from .configuration import TinyDocVLMConfig
from .vision_encoder import SigLIPVisionEncoder
from .token_compressor import PixelShuffleTokenCompressor
from .decoder import TinyDocDecoder
from .modeling import TinyDocVLMForConditionalGeneration, TinyDocVLMPreTrainedModel
from .image_processing import TinyDocImageProcessor
from .processing import TinyDocVLMProcessor

# Register configurations and models for HuggingFace Auto classes
from transformers import AutoConfig, AutoModelForCausalLM

try:
    AutoConfig.register("tinydoc_vlm", TinyDocVLMConfig)
    # We register it as AutoModelForCausalLM so AutoModelForVision2Seq etc can find it if needed
    # (usually AutoModelForCausalLM handles conditional generation)
    AutoModelForCausalLM.register(TinyDocVLMConfig, TinyDocVLMForConditionalGeneration)
except ValueError:
    # Already registered or failed due to another registration conflict
    pass
