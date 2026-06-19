from .configuration import TinyDocVLMConfig
from .vision_encoder import SigLIPVisionEncoder
from .token_compressor import PixelShuffleTokenCompressor
from .decoder import TinyDocDecoder
from .modeling import TinyDocVLMForConditionalGeneration, TinyDocVLMPreTrainedModel
from .image_processing import TinyDocImageProcessor
from .processing import TinyDocVLMProcessor
from .output_heads import MultiTaskOutputHeads, JSONHead, KVHead, TableHead, OCRHead, QAHead
from .data import DocumentDataset
from .losses import CombinedLoss
from .trainer import TinyDocVLMTrainer, TrainerConfig

__all__ = [
    "TinyDocVLMConfig",
    "SigLIPVisionEncoder",
    "PixelShuffleTokenCompressor",
    "TinyDocDecoder",
    "TinyDocVLMForConditionalGeneration",
    "TinyDocVLMPreTrainedModel",
    "TinyDocImageProcessor",
    "TinyDocVLMProcessor",
    "MultiTaskOutputHeads",
    "JSONHead",
    "KVHead",
    "TableHead",
    "OCRHead",
    "QAHead",
    "DocumentDataset",
    "CombinedLoss",
    "TinyDocVLMTrainer",
    "TrainerConfig",
]

from transformers import AutoConfig, AutoModelForCausalLM

try:
    AutoConfig.register("tinydoc_vlm", TinyDocVLMConfig)
    AutoModelForCausalLM.register(TinyDocVLMConfig, TinyDocVLMForConditionalGeneration)
except ValueError:
    pass
