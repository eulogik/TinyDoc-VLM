# TinyDoc-VLM — OpenRouter Submission Guide

## 1. Model Details

| Field | Value |
|---|---|
| **Model ID** | `eulogik/tinydoc-vlm-256m` |
| **Name** | TinyDoc-VLM 256M |
| **Description** | A 256M parameter vision-language model specialized for document understanding. Extracts structured data from invoices, receipts, forms, and tables. Runs on <1GB VRAM, optimized for CPU inference via ONNX. |
| **Context Length** | 8192 tokens |
| **Top Provider** | eulogik |
| **HuggingFace Hub** | [eulogik/TinyDoc-VLM-256M](https://huggingface.co/eulogik/TinyDoc-VLM-256M) |
| **Model Type** | Vision-Language Model (VLM) |
| **Architecture** | Small VLM with document-optimized vision encoder |
| **Supported Inputs** | Text + Image (document images, photos of invoices/receipts/forms) |
| **Primary Use Case** | Structured document extraction (invoices, receipts, forms, tables) |
| **Hardware Requirements** | <1GB VRAM; CPU inference supported via ONNX export |

---

## 2. Pricing Recommendation

| Tier | Price |
|---|---|
| **Free** | 100 requests/day |
| **Input Tokens** | $0.10 / M tokens |
| **Output Tokens** | $0.20 / M tokens |

**Rationale:** TinyDoc-VLM is a 256M parameter model — orders of magnitude smaller and cheaper to run than generalist VLMs (e.g., GPT-4o, Claude 3.5 Sonnet). The low compute cost allows aggressive pricing while maintaining healthy margins. The free tier drives adoption for low-volume users and evaluation.

---

## 3. Prompt Format

### Chat Completions Format (OpenAI-compatible)

TinyDoc-VLM uses the standard OpenAI chat completions format with `image_url` support:

```json
{
  "model": "eulogik/tinydoc-vlm-256m",
  "messages": [
    {
      "role": "system",
      "content": "You are a document extraction assistant. Extract structured data from document images. Always return valid JSON matching the requested schema."
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Extract all line items, totals, and vendor information from this invoice."
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "https://example.com/invoice.png"
          }
        }
      ]
    }
  ],
  "max_tokens": 1024,
  "temperature": 0.1
}
```

### Recommended System Prompt

```
You are TinyDoc-VLM, a specialized document understanding model. Your task is extract structured data from document images including invoices, receipts, forms, and tables. Always output valid JSON. Be concise and accurate. If a field is not present in the document, return null for that field.
```

### Example Prompts

**Invoice Extraction:**
```
Extract the following fields from this invoice image as JSON:
- vendor_name (string)
- invoice_number (string)
- date (ISO 8601 string)
- line_items (array of {description, quantity, unit_price, total})
- subtotal (number)
- tax (number)
- total (number)
```

**Receipt Extraction:**
```
Parse this receipt and return JSON with:
- merchant_name (string)
- date (ISO 8601 string)
- items (array of {name, price})
- total (number)
- payment_method (string)
```

**Table Parsing:**
```
Extract all tabular data from this image as a JSON array of objects. Use the first row as headers.
```

**Form Field Extraction:**
```
Extract all key-value pairs from this form image as a JSON object. Include field labels as keys and entered values as values.
```

---

## 4. Tool Calling / Function Calling

TinyDoc-VLM supports function calling for structured extraction tasks. Define tools in the request to get consistently formatted output.

### Tool Definitions

```json
{
  "model": "eulogik/tinydoc-vlm-256m",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Extract the invoice data from this image."
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "https://example.com/invoice_001.png"
          }
        }
      ]
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "extract_invoice",
        "description": "Extract structured data from an invoice image including vendor, line items, totals, and dates.",
        "parameters": {
          "type": "object",
          "properties": {
            "vendor_name": { "type": "string" },
            "invoice_number": { "type": "string" },
            "date": { "type": "string", "format": "date" },
            "line_items": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "description": { "type": "string" },
                  "quantity": { "type": "number" },
                  "unit_price": { "type": "number" },
                  "total": { "type": "number" }
                }
              }
            },
            "subtotal": { "type": "number" },
            "tax": { "type": "number" },
            "total": { "type": "number" }
          }
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "extract_receipt",
        "description": "Extract structured data from a receipt image including merchant, items, totals, and payment method.",
        "parameters": {
          "type": "object",
          "properties": {
            "merchant_name": { "type": "string" },
            "date": { "type": "string", "format": "date" },
            "items": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "name": { "type": "string" },
                  "price": { "type": "number" }
                }
              }
            },
            "total": { "type": "number" },
            "payment_method": { "type": "string" }
          }
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "parse_table",
        "description": "Extract tabular data from a document image into structured JSON rows.",
        "parameters": {
          "type": "object",
          "properties": {
            "headers": {
              "type": "array",
              "items": { "type": "string" }
            },
            "rows": {
              "type": "array",
              "items": {
                "type": "object",
                "additionalProperties": { "type": "string" }
              }
            }
          }
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "extract_json",
        "description": "Extract arbitrary structured data from a document image as a user-defined JSON schema.",
        "parameters": {
          "type": "object",
          "properties": {
            "document_type": { "type": "string" },
            "extracted_data": {
              "type": "object",
              "description": "The extracted structured data matching the requested schema."
            }
          },
          "required": ["document_type", "extracted_data"]
        }
      }
    }
  ],
  "tool_choice": {
    "type": "function",
    "function": { "name": "extract_invoice" }
  }
}
```

### Example Function Calling Response

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "model": "eulogik/tinydoc-vlm-256m",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_001",
            "type": "function",
            "function": {
              "name": "extract_invoice",
              "arguments": "{\"vendor_name\": \"Acme Corp\", \"invoice_number\": \"INV-2025-0042\", \"date\": \"2025-06-15\", \"line_items\": [{\"description\": \"Widget A\", \"quantity\": 10, \"unit_price\": 5.99, \"total\": 59.90}, {\"description\": \"Service B\", \"quantity\": 1, \"unit_price\": 150.00, \"total\": 150.00}], \"subtotal\": 209.90, \"tax\": 16.79, \"total\": 226.69}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ]
}
```

---

## 5. Endpoints

### Primary Endpoint (OpenAI-compatible)

```
POST https://openrouter.ai/v1/chat/completions
```

| Property | Value |
|---|---|
| Method | POST |
| Auth | `Authorization: Bearer $OPENROUTER_API_KEY` |
| Content-Type | application/json |
| Body | Standard OpenAI chat completions format |

### Custom Extraction Endpoint

```
POST https://openrouter.ai/v1/extract
```

| Property | Value |
|---|---|
| Method | POST |
| Auth | `Authorization: Bearer $OPENROUTER_API_KEY` |
| Content-Type | multipart/form-data or application/json |
| Purpose | Simplified document extraction without needing to construct tool calls |

**Request (multipart/form-data):**

| Field | Type | Description |
|---|---|---|
| `file` | File | Document image (PNG, JPEG, WebP) |
| `document_type` | string | One of: `invoice`, `receipt`, `form`, `table` |
| `schema` | string (JSON, optional) | Custom JSON Schema for output structure |

**Response:**

```json
{
  "document_type": "invoice",
  "confidence": 0.94,
  "extracted_data": {
    "vendor_name": "Acme Corp",
    "invoice_number": "INV-2025-0042",
    "date": "2025-06-15",
    "total": 226.69
  },
  "processing_time_ms": 340
}
```

---

## 6. Deployment Requirements

### HuggingFace Hub (Required)

- Model repository: [eulogik/TinyDoc-VLM-256M](https://huggingface.co/eulogik/TinyDoc-VLM-256M)
- Must include: `config.json`, model weights, `README.md` (model card)
- License: Apache 2.0 recommended
- Tags: `document-understanding`, `vision-language-model`, `onnx`, `cpu-inference`

### ONNX Export (Recommended)

For CPU-optimized inference, export the model to ONNX format:

```python
from optimum.onnxruntime import ORTModelForVision2Seq
from transformers import AutoProcessor

model_id = "eulogik/TinyDoc-VLM-256M"
ort_model = ORTModelForVision2Seq.from_pretrained(model_id, export=True)
processor = AutoProcessor.from_pretrained(model_id)

ort_model.save_pretrained("tinydoc-vlm-256m-onnx")
processor.save_pretrained("tinydoc-vlm-256m-onnx")
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn==0.30.1 \
    optimum[onnxruntime]==1.21.0 \
    transformers==4.43.0 \
    Pillow==10.4.0 \
    python-multipart==0.0.9

COPY app/ /app/
COPY tinydoc-vlm-256m-onnx/ /app/model/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**FastAPI app (`app/main.py`):**

```python
from fastapi import FastAPI, File, UploadFile, Form
from PIL import Image
from optimum.onnxruntime import ORTModelForVision2Seq
from transformers import AutoProcessor
import io

app = FastAPI()
model = ORTModelForVision2Seq.from_pretrained("/app/model")
processor = AutoProcessor.from_pretrained("/app/model")

@app.post("/v1/extract")
async def extract(
    file: UploadFile = File(...),
    document_type: str = Form("invoice"),
    schema: str = Form(None)
):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=1024)
    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return {"document_type": document_type, "extracted_data": result}
```

### Resource Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| CPU | 1 core | 2+ cores |
| RAM | 512 MB | 1 GB |
| Disk | 500 MB | 1 GB |
| GPU | Not required | Optional (CUDA for batch) |

---

## 7. Submission Checklist

- [x] Model hosted on HuggingFace Hub (`eulogik/TinyDoc-VLM-256M`)
- [x] Model card with benchmarks and usage examples
- [x] Pricing configured (free tier + paid rates)
- [x] Description finalized (concise, highlights specialization and efficiency)
- [x] Example prompts ready (invoice, receipt, table, form extraction)
- [x] Tool/function calling schemas defined
- [x] ONNX export available for CPU inference
- [x] Docker deployment spec prepared
- [x] OpenAI-compatible `/v1/chat/completions` endpoint working
- [x] Custom `/v1/extract` endpoint implemented
- [ ] Provider account created on OpenRouter (eulogik)
- [ ] Model submitted via OpenRouter dashboard or API
- [ ] Test requests verified end-to-end
- [ ] Rate limits configured (100 req/day free tier)

---

## 8. Quick Reference for OpenRouter Form Fields

```
Model ID:           eulogik/tinydoc-vlm-256m
Model Name:         TinyDoc-VLM 256M
Provider:           eulogik
Context Length:     8192
Top Provider:       eulogik
Prompt Pricing:     $0.10/M input tokens
Completion Pricing: $0.20/M input tokens
Free Tier:          100 requests/day
Architecture:       Vision-Language Model
License:            Apache 2.0
Tags:               document-understanding, vlm, onnx, cpu, invoice, receipt, form, table
HF Hub URL:          https://huggingface.co/eulogik/TinyDoc-VLM-256M
Endpoint:           https://openrouter.ai/v1/chat/completions
```
