import os, sys, torch, gradio as gr
from PIL import Image
from pathlib import Path

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent / "tinydoc_vlm"))
from tinydoc_vlm import TinyDocVLMForConditionalGeneration, TinyDocVLMProcessor

MODEL_ID = "eulogik/TinyDoc-VLM-256M"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[TinyDoc] Starting...", flush=True)
print(f"[TinyDoc] DEVICE={device} MODEL={MODEL_ID}", flush=True)

try:
    model = TinyDocVLMForConditionalGeneration.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )
    model.to(device).eval()
    print(f"[TinyDoc] Model loaded on {device}", flush=True)
except Exception as e:
    print(f"[TinyDoc] Model load FAILED: {e}", flush=True)
    raise

processor = TinyDocVLMProcessor()
print(f"[TinyDoc] Processor ready", flush=True)

def run(image, question, task):
    if image is None:
        return "Please upload a document image."
    prompt = f"<image>\n{'Answer: ' + question if task == 'Ask a question' else 'Extract JSON: ' if task == 'Extract JSON' else 'Convert table to Markdown: '}"
    inputs = processor(prompt, images=image)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    text = processor.tokenizer.decode(out[0], skip_special_tokens=True)
    return text

with gr.Blocks(title="TinyDoc-VLM — Document Understanding", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📄 TinyDoc-VLM
    ### The World's Smallest Document Understanding Model
    Upload a document image and ask questions or extract structured data.
    """)
    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(type="pil", label="Document Image", height=400)
            task = gr.Radio(["Ask a question", "Extract JSON", "Extract Table"], label="Task", value="Ask a question")
            question = gr.Textbox(label="Question", value="What is the total?", lines=2, visible=True)
            def update_question(task):
                return gr.update(visible=task == "Ask a question")
            task.change(fn=update_question, inputs=task, outputs=question)
            with gr.Row():
                submit = gr.Button("▶ Run", variant="primary", scale=2)
                clear = gr.Button("🗑 Clear", scale=1)
        with gr.Column(scale=1):
            output = gr.Markdown(label="Result")
    submit.click(fn=run, inputs=[image, question, task], outputs=output)
    def clear_all():
        return None, "Ask a question", "What is the total?", ""
    clear.click(fn=clear_all, outputs=[image, task, question, output])
    gr.Markdown("""
    ---
    **Model**: [eulogik/TinyDoc-VLM-256M](https://huggingface.co/eulogik/TinyDoc-VLM-256M) · 
    **Code**: [github.com/eulogik/TinyDoc-VLM](https://github.com/eulogik/TinyDoc-VLM) · 
    **By**: [eulogik](https://eulogik.com) · 
    [🐍 PyPI](https://pypi.org/project/tinydoc/) · 
    [🐦 @eulogik](https://twitter.com/eulogik)
    """)

if __name__ == "__main__":
    print(f"[TinyDoc] Starting Gradio...", flush=True)
    demo.launch(server_name="0.0.0.0", server_port=7860)
