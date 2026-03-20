[Medium Article](https://medium.com/@ai.omar.rehan/fine-tuning-qwen2-5-vl-for-astronomy-with-unsloth-a-compact-end-to-end-workflow-1e9bf9062d2d)

---

[Hugging Face Space](https://huggingface.co/spaces/AIOmarRehan/AstroVision-Fine-Tuning-Qwen2.5-VL)

---

# AstroVision Chat

AstroVision Chat is a Hugging Face Space for astronomy image understanding. It loads the `unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit` base model, applies the astronomy LoRA adapter, and generates responses from uploaded images through a Gradio interface.

## Tools Used

![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FF9D00?style=for-the-badge&logo=huggingface&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-F97316?style=for-the-badge&logo=gradio&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![PEFT](https://img.shields.io/badge/PEFT-4B5563?style=for-the-badge&logo=huggingface&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Qwen](https://img.shields.io/badge/Qwen-7C3AED?style=for-the-badge&logo=alibabacloud&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white)
![Unsloth](https://img.shields.io/badge/Unsloth-0EA5E9?style=for-the-badge)

## Model Setup

- Base model: `unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit`
- LoRA adapter: `AIOmarRehan/Qwen2_5_VL_7B-Vision-LoRA-on-Astronomy-with-Unsloth`
- App entrypoint: `app.py`

## How It Works

1. Upload an astronomy image.
2. Enter a prompt or use the default one.
3. The Space runs multimodal inference and returns a grounded response.

## Runtime Note

This Model is designed for a GPU runtime. The model is loaded in 4-bit mode, so inference requires CUDA.
