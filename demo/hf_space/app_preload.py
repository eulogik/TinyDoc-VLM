import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "tinydoc_vlm"))
from tinydoc_vlm import TinyDocVLMForConditionalGeneration
MODEL_ID = "eulogik/TinyDoc-VLM-256M"
print(f"Pre-downloading {MODEL_ID}...")
TinyDocVLMForConditionalGeneration.from_pretrained(MODEL_ID)
print("Model cached!")
