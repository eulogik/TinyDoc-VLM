
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM


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
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
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
