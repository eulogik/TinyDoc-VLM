import torch
from PIL import Image
import numpy as np

from tinydoc_vlm import (
    TinyDocVLMConfig,
    TinyDocVLMForConditionalGeneration,
    TinyDocImageProcessor,
    TinyDocVLMProcessor,
)

def test_config():
    config = TinyDocVLMConfig()
    assert config.pixel_shuffle_scale == 3
    assert config.image_size == 768
    assert config.patch_size == 16
    assert config.vision_config.hidden_size == 768
    assert config.decoder_config.hidden_size == 576

def test_image_processor():
    processor = TinyDocImageProcessor(image_size=768, tiling_mode="auto")
    
    # Check single tile preprocessing (small image)
    small_img = Image.fromarray(np.uint8(np.random.rand(200, 200, 3) * 255))
    small_out = processor.preprocess(small_img)
    assert small_out.shape == (1, 3, 768, 768)
    
    # Check multi-tile preprocessing (large image)
    large_img = Image.fromarray(np.uint8(np.random.rand(800, 600, 3) * 255))
    large_out = processor.preprocess(large_img)
    # 800/768 -> ceil is 2. But cols and rows are capped at 2.
    # cols = 1 (600/768 -> ceil is 1)
    # rows = 2 (800/768 -> ceil is 2, capped at 2)
    # Total tiles = 1 thumbnail + 1 * 2 = 3 tiles
    assert large_out.shape == (3, 3, 768, 768)

def test_model_forward():
    config = TinyDocVLMConfig()
    
    # Reduce size for faster unit tests
    config.vision_config.num_hidden_layers = 2
    config.decoder_config.num_hidden_layers = 2
    config.decoder_config.vocab_size = 1000
    config.image_token_id = 999
    
    model = TinyDocVLMForConditionalGeneration(config)
    
    # Dummy inputs
    B = 2
    N = 3 # 3 tiles
    C = 3
    H = W = 768
    
    pixel_values = torch.randn(B, N, C, H, W)
    # input_ids: must contain placeholder tokens (image_token_id)
    # For default 768 size, 16 patch, scale 3: (768/16/3)^2 = 16^2 = 256 tokens per tile.
    # Total visual tokens = 3 * 256 = 768 tokens.
    # So we need at least 768 placeholders. Let's make sequence length 800.
    seq_len = 800
    input_ids = torch.randint(0, 990, (B, seq_len))
    # Replace first 768 tokens with image_token_id (999) to act as placeholders
    input_ids[:, :768] = 999
    
    attention_mask = torch.ones(B, seq_len, dtype=torch.long)
    
    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask
    )
    
    assert outputs.logits.shape == (B, seq_len, 1000)

def test_processor_integration():
    config = TinyDocVLMConfig()
    # Reduce size for speed
    config.vision_config.num_hidden_layers = 1
    config.decoder_config.num_hidden_layers = 1
    
    model = TinyDocVLMForConditionalGeneration(config)
    processor = TinyDocVLMProcessor()
    
    # Resize model embeddings to match the extended tokenizer vocab
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    model.config.decoder_config.vocab_size = len(processor.tokenizer)
    # Use the processor's image_token_id in the model
    model.config.image_token_id = processor.image_token_id
    model.image_token_id = processor.image_token_id
    
    # Ensure pad token id is set
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    
    # Generate some dummy images
    img1 = Image.fromarray(np.uint8(np.random.rand(400, 400, 3) * 255))
    img2 = Image.fromarray(np.uint8(np.random.rand(200, 200, 3) * 255))
    
    # Texts with <image> placeholder
    texts = [
        "Extract details: <image>",
        "Describe document: <image>"
    ]
    
    # Process
    inputs = processor(text=texts, images=[img1, img2], padding=True)
    
    assert "input_ids" in inputs
    assert "pixel_values" in inputs
    assert "attention_mask" in inputs
    
    # Run forward pass through model using inputs
    outputs = model(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        attention_mask=inputs["attention_mask"]
    )
    
    assert outputs.logits.shape[0] == 2  # batch size
