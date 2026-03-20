import importlib
import os
from datetime import datetime

import gradio as gr
import torch

try:
    import spaces
except Exception:
    class _SpacesFallback:
        @staticmethod
        def GPU(duration=180):
            def _decorator(function):
                return function
            return _decorator

    spaces = _SpacesFallback()


BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit")
LORA_REPO_ID = os.getenv(
    "LORA_REPO_ID",
    "AIOmarRehan/Qwen2_5_VL_7B-Vision-LoRA-on-Astronomy-with-Unsloth",
)
DEFAULT_PROMPT = "You are an expert astronomer. Describe accurately what you see in this image."

model = None
tokenizer = None
model_load_time = None


def load_model():
    """Load the exact notebook deployment stack: Unsloth base model + LoRA adapter."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU is not currently available. This Space can boot on CPU, but the exact Unsloth Qwen2.5-VL 7B + LoRA inference path needs GPU hardware."
        )

    FastVisionModel = importlib.import_module("unsloth").FastVisionModel

    loaded_model, loaded_tokenizer = FastVisionModel.from_pretrained(
        BASE_MODEL_ID,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )

    loaded_model = FastVisionModel.get_peft_model(loaded_model, lora_adapter=LORA_REPO_ID)
    loaded_model.eval()
    FastVisionModel.for_inference(loaded_model)

    return loaded_model, loaded_tokenizer


def ensure_model_loaded():
    global model, tokenizer, model_load_time
    if model is None or tokenizer is None:
        model, tokenizer = load_model()
        model_load_time = datetime.now()


def clean_response(text: str) -> str:
    if "assistant" in text:
        text = text.split("assistant")[-1]
    return text.strip()


@spaces.GPU(duration=180)
def describe_astronomy_image(image, custom_prompt=None, temperature=1.2, max_tokens=256, top_p=0.9):
    if image is None:
        return "Please upload an image to analyze."

    try:
        ensure_model_loaded()

        prompt = custom_prompt.strip() if custom_prompt and custom_prompt.strip() else DEFAULT_PROMPT
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                do_sample=True,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return clean_response(generated_text)
    except Exception as error:
        error_text = str(error).lower()
        if "cuda" in error_text or "gpu" in error_text:
            return (
                "GPU is not detected at inference time. The app is using the exact notebook artifacts "
                "and can start on CPU, but actual inference requires Space hardware set to GPU."
            )
        return f"Error: {error}"


custom_css = """
:root {
    --primary-color: #1f77b4;
    --secondary-color: #667eea;
    --accent-color: #764ba2;
}

#header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 30px;
    border-radius: 10px;
    margin-bottom: 20px;
    text-align: center;
}

#header h1 {
    margin: 0;
    font-size: 2.5em;
}

#header p {
    margin: 10px 0 0 0;
    font-size: 1.05em;
    opacity: 0.95;
}
"""


with gr.Blocks(title="AstroVision Chat", css=custom_css, theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")) as demo:
    gr.HTML(
        """
        <div id="header">
            <h1>AstroVision Chat</h1>
            <p>Exact notebook deployment stack: Unsloth + Qwen2.5-VL 7B + astronomy LoRA</p>
        </div>
        """
    )

    runtime_mode = "GPU enabled" if torch.cuda.is_available() else "CPU startup only"
    gr.Markdown(
        f"**Runtime mode:** {runtime_mode}. This app keeps the exact notebook artifacts. "
        "On free CPU Spaces the UI loads normally, but inference will only work after switching the Space to GPU."
    )

    with gr.Tabs():
        with gr.TabItem("Analyzer", id="analyzer"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        label="Upload Astronomy Image",
                        type="pil",
                        sources=["upload", "webcam"],
                        interactive=True,
                    )

                with gr.Column(scale=1):
                    prompt_input = gr.Textbox(
                        label="Custom Prompt",
                        placeholder=DEFAULT_PROMPT,
                        lines=3,
                    )

                    with gr.Accordion("Advanced Settings", open=False):
                        temperature_slider = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=1.2,
                            step=0.1,
                            label="Temperature",
                        )
                        max_tokens_slider = gr.Slider(
                            minimum=64,
                            maximum=512,
                            value=256,
                            step=32,
                            label="Max Response Length",
                        )
                        top_p_slider = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.9,
                            step=0.05,
                            label="Top-P",
                        )

            output = gr.Textbox(label="Model Output", lines=12, interactive=False)

            with gr.Row():
                submit_btn = gr.Button("Analyze Image", variant="primary", size="lg", scale=2)
                clear_btn = gr.Button("Clear All", variant="secondary", scale=1)

            submit_btn.click(
                fn=describe_astronomy_image,
                inputs=[image_input, prompt_input, temperature_slider, max_tokens_slider, top_p_slider],
                outputs=output,
            )

            clear_btn.click(
                lambda: (None, "", ""),
                outputs=[image_input, prompt_input, output],
            )

        with gr.TabItem("About Model", id="about"):
            loaded_at = model_load_time.strftime("%Y-%m-%d %H:%M:%S") if model_load_time else "Not loaded yet"
            gr.Markdown(
                f"""
                ## Exact Notebook Stack

                - Base model: `{BASE_MODEL_ID}`
                - LoRA adapter: `{LORA_REPO_ID}`
                - Loader: `FastVisionModel.from_pretrained(...)`
                - Adapter merge: `FastVisionModel.get_peft_model(..., lora_adapter=...)`
                - Inference mode: `FastVisionModel.for_inference(model)`
                - Model load time: `{loaded_at}`

                This matches the Qwen deployment notebook path. The model is lazy-loaded, so the Space can start on CPU without importing Unsloth during boot.
                """
            )

    gr.Markdown(
        """
        ---
        **Model:** Qwen2.5-VL 7B + LoRA | **Runtime behavior:** CPU-safe startup, GPU-required inference
        """
    )


if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )