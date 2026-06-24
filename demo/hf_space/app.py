import os, sys, torch, gradio as gr
import traceback
from PIL import Image
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent / "tinydoc_vlm"))
from tinydoc_vlm import TinyDocVLMForConditionalGeneration, TinyDocVLMProcessor

MODEL_ID = "eulogik/TinyDoc-VLM-256M"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[TinyDoc] Starting on {device}...", flush=True)

try:
    model = TinyDocVLMForConditionalGeneration.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )
    model.to(device).eval()
    processor = TinyDocVLMProcessor()
    # Sync image token ID between processor and model
    model.image_token_id = processor.image_token_id
    print(f"[TinyDoc] Model loaded! image_token_id={processor.image_token_id}", flush=True)
except Exception as e:
    print(f"[TinyDoc] LOAD ERROR: {e}", flush=True)
    traceback.print_exc()
    raise

def run(image, question, task):
    try:
        print(f"[TinyDoc] run() called: task={task}, question={question}, image_type={type(image)}", flush=True)
        
        if image is None:
            return "Please upload a document image."
        
        # Handle different image types from Gradio
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, dict):
            if "path" in image:
                image = Image.open(image["path"]).convert("RGB")
            elif "url" in image:
                import requests
                from io import BytesIO
                resp = requests.get(image["url"], timeout=10)
                image = Image.open(BytesIO(resp.content)).convert("RGB")
        elif hasattr(image, "convert"):
            image = image.convert("RGB")
        else:
            print(f"[TinyDoc] Unknown image type: {type(image)}", flush=True)
            return f"Error: Unknown image type {type(image)}"
        
        print(f"[TinyDoc] Image size: {image.size}", flush=True)
        
        if task == "Ask a question":
            prompt = f"<image>\nAnswer: {question}"
        elif task == "Extract JSON":
            prompt = "<image>\nExtract JSON: "
        else:
            prompt = "<image>\nConvert table to Markdown: "
        
        print(f"[TinyDoc] Prompt: {prompt[:80]}...", flush=True)
        
        inputs = processor(prompt, images=image)
        # Remove non-model kwargs before generate
        inputs.pop("image_token_id", None)
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        print(f"[TinyDoc] Running inference...", flush=True)
        
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        
        text = processor.tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"[TinyDoc] Output: {text[:100]}...", flush=True)
        return text
        
    except Exception as e:
        error_msg = f"Error: {e}\n{traceback.format_exc()}"
        print(f"[TinyDoc] RUN ERROR: {error_msg}", flush=True)
        return f"Error: {e}"

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
    print("[TinyDoc] Starting Gradio server...", flush=True)
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
