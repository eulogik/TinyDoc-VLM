from typing import Dict, Any, Union
from transformers import PretrainedConfig, AutoConfig

class TinyDocVLMConfig(PretrainedConfig):
    model_type = "tinydoc_vlm"
    is_composition = True

    def __init__(
        self,
        vision_config: Union[Dict[str, Any], PretrainedConfig] = None,
        decoder_config: Union[Dict[str, Any], PretrainedConfig] = None,
        pixel_shuffle_scale: int = 3,
        image_size: int = 384,
        patch_size: int = 16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # Set defaults if not provided
        if vision_config is None:
            # Default SigLIP-B/16-like configuration (approx 93M parameters)
            vision_config = {
                "model_type": "siglip_vision_model",
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "patch_size": 16,
                "image_size": 384,
                "num_channels": 3,
                "layer_norm_eps": 1e-6,
            }
            
        if decoder_config is None:
            # Default SmolLM2-135M-like configuration (approx 135M parameters)
            decoder_config = {
                "model_type": "llama",
                "vocab_size": 49152,
                "hidden_size": 576,
                "intermediate_size": 1536,
                "num_hidden_layers": 30,
                "num_attention_heads": 9,
                "num_key_value_heads": 3,
                "max_position_embeddings": 8192,
                "rms_norm_eps": 1e-5,
                "rope_theta": 273000.0,
                "attention_bias": False,
            }

        # Initialize config objects
        if isinstance(vision_config, dict):
            vision_config_copy = vision_config.copy()
            vision_model_type = vision_config_copy.pop("model_type", "siglip_vision_model")
            self.vision_config = AutoConfig.for_model(vision_model_type, **vision_config_copy)
        else:
            self.vision_config = vision_config

        if isinstance(decoder_config, dict):
            decoder_config_copy = decoder_config.copy()
            decoder_model_type = decoder_config_copy.pop("model_type", "llama")
            self.decoder_config = AutoConfig.for_model(decoder_model_type, **decoder_config_copy)
        else:
            self.decoder_config = decoder_config

        self.pixel_shuffle_scale = pixel_shuffle_scale
        self.image_size = image_size
        self.patch_size = patch_size

    def to_dict(self) -> Dict[str, Any]:
        output = super().to_dict()
        output["vision_config"] = self.vision_config.to_dict()
        output["decoder_config"] = self.decoder_config.to_dict()
        return output
