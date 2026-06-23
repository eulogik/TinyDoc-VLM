import gradio as gr
import torch
from PIL import Image
from tinydoc import TinyDocExtractor

extractor = TinyDocExtractor(model_name_or_id="eulogik/TinyDoc-VLM-256M", device="cuda")

def process(image, question, task):
    if image is None:
        return "Please upload a document image."
    if task == "Ask a question":
        res = extractor.ask(image, question)
        return f"**Answer:** {res.answer}\n\n*Latency: {res.latency_ms:.0f}ms, Tokens: {res.num_tokens_generated}*"
    elif task == "Extract JSON":
        res = extractor.extract(image, output_format="json")
        return f"**Raw text:**\n{res.raw_text}\n\n**Fields:**\n{res.fields}"
    elif task == "Extract Table":
        res = extractor.extract_table(image)
        return f"**Markdown:**\n{res.markdown}\n\n**Raw:**\n{res.raw_table}"
    return "Select a task."

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
    submit.click(fn=process, inputs=[image, question, task], outputs=output)
    def clear_all():
        return None, "Ask a question", "What is the total?", ""
    clear.click(fn=clear_all, outputs=[image, task, question, output])
    gr.Markdown("""
    ---
    **Model**: [eulogik/TinyDoc-VLM-256M](https://huggingface.co/eulogik/TinyDoc-VLM-256M) · 
    **Code**: [github.com/eulogik/TinyDoc-VLM](https://github.com/eulogik/TinyDoc-VLM) · 
    **By**: [eulogik](https://eulogik.com)
    """)
