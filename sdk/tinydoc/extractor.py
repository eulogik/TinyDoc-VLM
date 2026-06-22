import time
import re
import json
import logging
from pathlib import Path
from typing import Union, Dict, Any, Optional, List

from PIL import Image
import torch
import numpy as np

# Try to import from tinydoc_vlm package
try:
    from tinydoc_vlm import (
        TinyDocVLMForConditionalGeneration,
        TinyDocVLMProcessor,
        TinyDocVLMConfig
    )
except ImportError:
    # If running from inside the repository and not installed as package
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from tinydoc_vlm import (
        TinyDocVLMForConditionalGeneration,
        TinyDocVLMProcessor,
        TinyDocVLMConfig
    )

from .models import ExtractionResult, QAResult, TableResult

logger = logging.getLogger(__name__)

class TinyDocExtractor:
    """
    Python SDK for TinyDoc-VLM document extraction and VQA.
    Provides simple, one-liner APIs for document understanding tasks.
    Supports PyTorch and ONNX Runtime backends.
    """
    
    def __init__(
        self,
        model_path_or_id: str = "eulogik/TinyDoc-VLM-256M",
        device: Optional[str] = None,
        use_onnx: bool = False,
        onnx_model_path: Optional[str] = None
    ):
        self.use_onnx = use_onnx or (onnx_model_path is not None)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Load processor
        logger.info(f"Loading processor from {model_path_or_id}...")
        try:
            self.processor = TinyDocVLMProcessor.from_pretrained(model_path_or_id)
        except Exception as e:
            logger.warning(f"Could not load processor from {model_path_or_id} ({e}). Initialising processor with default configuration.")
            self.processor = TinyDocVLMProcessor()
            
        if self.use_onnx:
            import onnxruntime as ort
            self.onnx_path = onnx_model_path or str(Path(model_path_or_id) / "model.onnx")
            logger.info(f"Loading ONNX model from {self.onnx_path}...")
            self.session = ort.InferenceSession(self.onnx_path)
            self.model = None
        else:
            logger.info(f"Loading PyTorch model from {model_path_or_id}...")
            try:
                self.model = TinyDocVLMForConditionalGeneration.from_pretrained(model_path_or_id)
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                logger.warning(
                    f"Could not load PyTorch weights from {model_path_or_id} ({e}). "
                    "Creating a randomly initialized model for development/testing."
                )
                config = TinyDocVLMConfig()
                self.model = TinyDocVLMForConditionalGeneration(config)
                self.model.decoder.resize_token_embeddings(len(self.processor.tokenizer))
                self.model.to(self.device)
                self.model.eval()

    def _load_image(self, image_or_pdf: Union[str, Path, Image.Image], page: int = 1) -> Image.Image:
        """Helper to load input file/image into a PIL Image."""
        if isinstance(image_or_pdf, Image.Image):
            return image_or_pdf.convert("RGB")
            
        path = Path(image_or_pdf)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
            
        if path.suffix.lower() == ".pdf":
            try:
                from pdf2image import convert_from_path
                images = convert_from_path(path, first_page=page, last_page=page)
                if not images:
                    raise ValueError(f"Could not extract page {page} from PDF: {path}")
                return images[0].convert("RGB")
            except ImportError:
                raise ImportError("Please install pdf2image (`pip install pdf2image`) and ensure poppler is installed to read PDFs.")
        else:
            return Image.open(path).convert("RGB")

    def _generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2
    ) -> tuple[str, float, int]:
        """Runs autoregressive text generation using PyTorch or ONNX."""
        start_time = time.time()
        
        # Preprocess text and image
        inputs = self.processor(text=prompt, images=image, padding=True)
        
        # Extracted shapes
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs.get("pixel_values")
        
        generated_tokens = 0
        
        if self.use_onnx:
            # Greedy generation loop using ONNX
            eos_token_id = self.processor.tokenizer.eos_token_id
            
            for _ in range(max_new_tokens):
                # Format ONNX input feed
                feed = {
                    "input_ids": input_ids.numpy(),
                    "attention_mask": attention_mask.numpy()
                }
                if pixel_values is not None:
                    feed["pixel_values"] = pixel_values.numpy()
                    
                outputs = self.session.run(None, feed)
                logits = outputs[0]  # Shape: (1, seq_len, vocab_size)
                
                # Next token prediction (greedy)
                next_token = int(np.argmax(logits[0, -1, :]))
                generated_tokens += 1
                
                if next_token == eos_token_id:
                    break
                    
                # Append next token
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.tensor([[1]])], dim=-1)
                
            output_text = self.processor.tokenizer.decode(
                input_ids[0, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
        else:
            # PyTorch generation
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            if pixel_values is not None:
                pixel_values = pixel_values.to(self.device)
                
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0.0,
                    temperature=temperature if temperature > 0.0 else None,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )
                
            generated_tokens = outputs.shape[1] - input_ids.shape[1]
            output_text = self.processor.tokenizer.decode(
                outputs[0, input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
        latency_ms = (time.time() - start_time) * 1000
        return output_text.strip(), latency_ms, generated_tokens

    def extract(
        self,
        image_or_pdf: Union[str, Path, Image.Image],
        output_format: str = "json",
        page: int = 1
    ) -> ExtractionResult:
        """
        Extracts document information as structured JSON fields.
        """
        img = self._load_image(image_or_pdf, page=page)
        
        prompt = "<image>\nExtract all fields as JSON."
        output_text, latency_ms, gen_tokens = self._generate(img, prompt)
        
        # Attempt to parse JSON from output text
        fields = {}
        # Clean any markdown block formatting (e.g. ```json ... ```)
        cleaned = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", output_text, flags=re.DOTALL).strip()
        
        try:
            fields = json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback regex search for JSON block
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    fields = json.loads(match.group(0))
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON structure from model output.")
            else:
                logger.warning("No JSON structure found in model output.")
                
        return ExtractionResult(
            raw_text=output_text,
            fields=fields,
            latency_ms=latency_ms,
            num_tokens_generated=gen_tokens
        )

    def ask(
        self,
        image_or_pdf: Union[str, Path, Image.Image],
        question: str,
        page: int = 1
    ) -> QAResult:
        """
        Asks a question about the document image or PDF.
        """
        img = self._load_image(image_or_pdf, page=page)
        prompt = f"<image>\nAnswer the following question about this document: {question}"
        
        output_text, latency_ms, gen_tokens = self._generate(img, prompt)
        
        return QAResult(
            question=question,
            answer=output_text,
            latency_ms=latency_ms,
            num_tokens_generated=gen_tokens
        )

    def extract_table(
        self,
        image_or_pdf: Union[str, Path, Image.Image],
        page: int = 1,
        format: str = "markdown"
    ) -> TableResult:
        """
        Extracts table structures from document pages and converts to Markdown or HTML.
        """
        img = self._load_image(image_or_pdf, page=page)
        prompt = "<image>\nConvert this table to HTML markup."
        
        output_text, latency_ms, gen_tokens = self._generate(img, prompt)
        
        markdown_table = self._html_table_to_markdown(output_text)
        
        return TableResult(
            raw_table=output_text,
            markdown=markdown_table,
            latency_ms=latency_ms,
            num_tokens_generated=gen_tokens
        )

    def _html_table_to_markdown(self, html: str) -> str:
        """Convert basic HTML table representation to markdown."""
        rows = re.findall(r"<tr>(.*?)</tr>", html, re.DOTALL)
        if not rows:
            return html
            
        md_rows = []
        headers = []
        
        # Try to extract headers from the first row
        header_matches = re.findall(r"<t[dh]>(.*?)</t[dh]>", rows[0], re.DOTALL)
        if header_matches:
            headers = [h.strip() for h in header_matches]
            md_rows.append("| " + " | ".join(headers) + " |")
            md_rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
            
        for row in rows[1:]:
            cell_matches = re.findall(r"<t[dh]>(.*?)</t[dh]>", row, re.DOTALL)
            if cell_matches:
                cells = [c.strip() for c in cell_matches]
                if headers:
                    if len(cells) < len(headers):
                        cells += [""] * (len(headers) - len(cells))
                    elif len(cells) > len(headers):
                        cells = cells[:len(headers)]
                md_rows.append("| " + " | ".join(cells) + " |")
                
        return "\n".join(md_rows)
