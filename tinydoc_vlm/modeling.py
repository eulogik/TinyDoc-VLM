from typing import Any

import torch
from torch import nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration import TinyDocVLMConfig
from .decoder import TinyDocDecoder
from .token_compressor import PixelShuffleTokenCompressor
from .vision_encoder import SigLIPVisionEncoder


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

class TinyDocVLMForConditionalGeneration(TinyDocVLMPreTrainedModel, GenerationMixin):
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
        self.image_token_id = getattr(config, "image_token_id", 49152)
        
        # 2D Positional Embeddings for visual features (added to tokens before projection)
        s = config.pixel_shuffle_scale
        compressed_grid_size = (config.image_size // config.patch_size) // s
        compressed_patches = compressed_grid_size ** 2
        
        # Learnable 2D positional embeddings for the compressed visual tokens
        self.visual_pos_embed = nn.Parameter(
            torch.zeros(1, 1, compressed_patches, config.decoder_config.hidden_size)
        )
        
        # Initialize weights
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.decoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.decoder.lm.set_input_embeddings(value)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        task: str | None = None,
    ) -> tuple | dict | CausalLMOutputWithPast:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Decoding pass (no new visual input, reuse cached states)
        if pixel_values is None and past_key_values is not None:
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
            return outputs

        # Prefill pass: merge text and visual tokens into inputs_embeds
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
                    # Vision features can come out fp32 (e.g. SigLIP LayerNorm
                    # stays fp32 under autocast); align with the embedding
                    # dtype before the index-put or this crashes on fp16/bf16
                    # weight checkpoints.
                    features_to_insert = flat_visual_features[b][:num_places].to(
                        inputs_embeds.dtype)
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
        
        return outputs

    def generate(self, *args, **kwargs):
        """
        Overrides GenerationMixin.generate to inject anti-repetition defaults
        (no_repeat_ngram_size + repetition_penalty), matching the long-horizon
        decoding trick used by SOTA OCR models (e.g. Baidu Unlimited-OCR).
        """
        kwargs.setdefault("no_repeat_ngram_size", 20)
        kwargs.setdefault("repetition_penalty", 1.1)
        return super().generate(*args, **kwargs)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        pixel_values=None,
        **kwargs
    ) -> dict[str, Any]:
        """
        Overridden to support KV caching during auto-regressive generation.
        """
        is_decoding = past_key_values is not None and pixel_values is None

        if is_decoding:
            input_ids = input_ids[:, -1:]
            inputs_embeds = None
            
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if is_decoding:
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
