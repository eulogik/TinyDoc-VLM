import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Union
from transformers import PreTrainedModel, LlamaConfig, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration import TinyDocVLMConfig
from .vision_encoder import SigLIPVisionEncoder
from .token_compressor import PixelShuffleTokenCompressor
from .decoder import TinyDocDecoder
from .output_heads import MultiTaskOutputHeads
from .attention import get_2d_sincos_pos_embed

class TinyDocVLMPreTrainedModel(PreTrainedModel):
    config_class = TinyDocVLMConfig
    base_model_prefix = "tinydoc_vlm"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class TinyDocVLMForConditionalGeneration(TinyDocVLMPreTrainedModel):
    """
    TinyDoc-VLM: The World's Smallest Document Understanding Model.
    Coordinates SigLIP Vision Encoder, PixelShuffle Compressor, and SmolLM2 Decoder.
    """
    def __init__(self, config: TinyDocVLMConfig):
        super().__init__(config)
        
        # 1. Vision Encoder
        self.vision_encoder = SigLIPVisionEncoder(config)
        
        # 2. Token Compressor / Connector
        self.compressor = PixelShuffleTokenCompressor(
            config, 
            encoder_dim=config.vision_config.hidden_size, 
            decoder_dim=config.decoder_config.hidden_size
        )
        
        # 3. Decoder
        self.decoder = TinyDocDecoder(config.decoder_config)
        
        # Learnable image pad / placeholder token ID
        self.image_token_id = getattr(config, "image_token_id", 49153)
        
        # 2D Positional Embeddings for visual features (added to tokens before projection)
        num_patches = (config.image_size // config.patch_size) ** 2
        s = config.pixel_shuffle_scale
        compressed_grid_size = (config.image_size // config.patch_size) // s
        compressed_patches = compressed_grid_size ** 2
        
        # Learnable 2D positional embeddings for the compressed visual tokens
        self.visual_pos_embed = nn.Parameter(
            torch.zeros(1, 1, compressed_patches, config.decoder_config.hidden_size)
        )
        
        # 4. Structured Output Heads (multi-task)
        self.output_heads = MultiTaskOutputHeads(
            hidden_size=config.decoder_config.hidden_size,
            vocab_size=config.decoder_config.vocab_size,
        )
        
        # Initialize weights
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.decoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.decoder.lm.set_input_embeddings(value)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        task: Optional[str] = None,
    ) -> Union[Tuple, Dict, CausalLMOutputWithPast]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if task else output_hidden_states
        
        # If we already have past_key_values, we do not need to process pixel_values again
        if past_key_values is not None:
            outputs = self.decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            if task:
                hidden = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else outputs[2]
                head_outputs = self.output_heads(hidden, task=task)
                return {"lm_outputs": outputs, "head_outputs": head_outputs}
            return outputs

        # First pass: construct inputs_embeds by merging text and visual tokens
        if inputs_embeds is None:
            inputs_embeds = self.decoder.get_input_embeddings()(input_ids)
            
        if pixel_values is not None:
            visual_features = self.vision_encoder(pixel_values)
            compressed_features = self.compressor(visual_features)
            compressed_features = compressed_features + self.visual_pos_embed
            
            batch_size, num_tiles, compressed_patches, decoder_dim = compressed_features.shape
            flat_visual_features = compressed_features.view(
                batch_size, num_tiles * compressed_patches, decoder_dim
            )
            
            image_mask = (input_ids == self.image_token_id)
            for b in range(batch_size):
                num_places = image_mask[b].sum().item()
                if num_places > 0:
                    features_to_insert = flat_visual_features[b][:num_places]
                    inputs_embeds[b, image_mask[b]] = features_to_insert

        outputs = self.decoder(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        if task:
            hidden = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else outputs[-1]
            head_outputs = self.output_heads(hidden, task=task)
            return {"lm_outputs": outputs, "head_outputs": head_outputs}
        
        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        pixel_values=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Overridden to support KV caching during auto-regressive generation.
        """
        # If we have past_key_values, we only need the last generated token
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            # We don't need inputs_embeds or pixel_values anymore as they are stored in the cache
            inputs_embeds = None
            pixel_values = None
            
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # Create position_ids on the fly if needed
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -input_ids.shape[-1]:]

        return {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "past_key_values": past_key_values,
            "pixel_values": pixel_values,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": kwargs.get("use_cache"),
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        return self.decoder.lm._reorder_cache(past_key_values, beam_idx)
