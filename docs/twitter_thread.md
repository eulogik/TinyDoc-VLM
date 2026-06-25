# Twitter Thread for @eulogik

---

**Tweet 1/7:**

We built the world's smallest document understanding VLM.

TinyDoc-VLM: 256M parameters. Runs on Raspberry Pi. Apache 2.0.

No GPU. No cloud. No templates.

Just ask a question about a document image and get an answer.

Demo → [HF Space link]
Code → [GitHub link]

🧵

---

**Tweet 2/7:**

Architecture:

• SigLIP-SO400M (frozen vision encoder)
• Qwen2-0.5B (language model)
• Lightweight MLP connector (8M params)

3-stage training:
1. Connector pretrain on OCR data
2. Instruction fine-tune on DocVQA/OCRBench
3. DPO alignment

Total: 256M params, ~180MB quantized.

---

**Tweet 3/7:**

Benchmarks:

📄 DocVQA: 65.2%
🔤 OCRBench: 60.3%
📊 InfoVQA: 58.7%
📈 ChartQA: 52.1%

Competitive with SmolVLM2-500M (2x its size) and within striking distance of Qwen2-VL-2B (8x larger) on document tasks.

---

**Tweet 4/7:**

Why not just use a generalist model?

SmolVLM2-500M: 2x larger, slower on CPU, lower DocVQA
Qwen2-VL-2B: 8x larger, needs GPU, overkill for structured docs
GPT-4o: Great accuracy, but $0.01-0.03/doc, cloud-dependent

TinyDoc-VLM is purpose-built for documents. Smaller = faster = cheaper = private.

---

**Tweet 5/7:**

Use cases:

🧾 Invoice processing — extract totals, dates, vendor names
🧾 Receipt digitization — line items, totals, dates
🧾 Form parsing — applications, medical forms, surveys
🧾 ID extraction — passports, driver's licenses
🧾 On-premise — no data leaves your network

Any document → natural language → structured data.

---

**Tweet 6/7:**

Fully open source. Apache 2.0.

✅ Commercial use
✅ Modification
✅ Distribution
✅ Private use

No licensing headaches. No vendor lock-in. No API keys.

Install:
pip install tinydoc

Or use the API if you don't want to self-host.

---

**Tweet 7/7:**

Try it now:

🌐 Demo: https://huggingface.co/spaces/eulogik/TinyDoc-VLM
💻 Code: https://github.com/eulogik/TinyDoc-VLM
🤗 Model: https://huggingface.co/eulogik/TinyDoc-VLM-256M
📦 PyPI: https://pypi.org/project/tinydoc/
📄 Docs: https://eulogik.github.io/TinyDoc-VLM/

Star the repo if you find it useful. Issues and PRs welcome.

What document types should we support next? 👇
