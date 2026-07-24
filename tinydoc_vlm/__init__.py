from .configuration import TinyDocVLMConfig
from .data import DocumentDataset
from .decoder import TinyDocDecoder
from .image_processing import TinyDocImageProcessor
from .losses import CombinedLoss
from .modeling import TinyDocVLMForConditionalGeneration, TinyDocVLMPreTrainedModel
from .processing import TinyDocVLMProcessor
from .token_compressor import PixelShuffleTokenCompressor
from .trainer import TinyDocVLMTrainer, TrainerConfig
from .vision_encoder import SigLIPVisionEncoder

__all__ = [
    "CombinedLoss",
    "DocumentDataset",
    "PixelShuffleTokenCompressor",
    "SigLIPVisionEncoder",
    "TinyDocDecoder",
    "TinyDocImageProcessor",
    "TinyDocVLMConfig",
    "TinyDocVLMForConditionalGeneration",
    "TinyDocVLMPreTrainedModel",
    "TinyDocVLMProcessor",
    "TinyDocVLMTrainer",
    "TrainerConfig",
]

from transformers import AutoConfig, AutoModelForCausalLM

try:
    AutoConfig.register("tinydoc_vlm", TinyDocVLMConfig)
    AutoModelForCausalLM.register(TinyDocVLMConfig, TinyDocVLMForConditionalGeneration)
except ValueError:
    pass
