import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from transformers import LlamaForCausalLM, LlamaConfig, PreTrainedModel

class TinyDocDecoder(nn.Module):
    """
    Decoder wrapper around LlamaForCausalLM (used by SmolLM2).
    Manages loading and vocabulary/embedding resizing for special tokens.
    """
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.lm = LlamaForCausalLM(config)
        self.hidden_size = config.hidden_size

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        """
        Resizes input token embeddings and output LM head of the decoder.
        """
        resized = self.lm.resize_token_embeddings(new_num_tokens)
        self.config.vocab_size = new_num_tokens
        return resized

    def get_input_embeddings(self) -> nn.Module:
        return self.lm.get_input_embeddings()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return self.lm(
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
