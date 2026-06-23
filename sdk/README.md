# TinyDoc Python SDK

`pip install tinydoc`

Python SDK for [TinyDoc-VLM](https://github.com/eulogik/TinyDoc-VLM) — the world's smallest document-specialist VLM by [eulogik](https://eulogik.com).

```python
from PIL import Image
from tinydoc import TinyDocExtractor

extractor = TinyDocExtractor()
img = Image.open("invoice.png")

result = extractor.ask(img, "What is the total?")
print(result.answer)

result = extractor.extract(img, output_format="json")
print(result.fields)

result = extractor.extract_table(img)
print(result.markdown)
```

[Documentation](https://github.com/eulogik/TinyDoc-VLM#readme) · [HF Model](https://huggingface.co/eulogik/TinyDoc-VLM-256M) · [Issues](https://github.com/eulogik/TinyDoc-VLM/issues)
