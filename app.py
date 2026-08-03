"""Gradio demo: upload a car photo, get top-5 predicted brand + confidence.

Runs locally with `python app.py`, and is the entrypoint for the Hugging Face Space
(the Space just needs this file + src/ + requirements.txt + a checkpoint in outputs/).
"""
import os
from pathlib import Path

import gradio as gr
import torch

from src.data import build_transforms
from src.model import build_model

MODEL_NAME = os.environ.get("CAR_MODEL", "resnet18")
IMAGE_SIZE = int(os.environ.get("CAR_IMAGE_SIZE", "128"))
CHECKPOINT_PATH = Path(os.environ.get("CAR_CHECKPOINT", f"outputs/{MODEL_NAME}_best.pth"))
CLASS_NAMES = ["Audi", "BMW", "Mercedes", "Porsche", "Volkswagen"]

device = "cuda" if torch.cuda.is_available() else "cpu"
transform = build_transforms(IMAGE_SIZE, train=False)

model = build_model(MODEL_NAME, num_classes=len(CLASS_NAMES), input_size=IMAGE_SIZE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.to(device).eval()


@torch.inference_mode()
def predict(image):
    if image is None:
        return None
    tensor = transform(image).unsqueeze(0).to(device)
    probs = torch.softmax(model(tensor), dim=1)[0].cpu()
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Car photo"),
    outputs=gr.Label(num_top_classes=5, label="Predicted brand"),
    title="Car Brand Classifier",
    description=(
        "Upload a photo of a car and the model predicts its brand "
        f"(Audi, BMW, Mercedes, Porsche, Volkswagen). Model: {MODEL_NAME}, "
        "fine-tuned/trained on ~2,500 scraped images. See the README for training "
        "curves, confusion matrix, and methodology."
    ),
)

if __name__ == "__main__":
    # ssr_mode=False: Gradio's Node.js SSR subprocess crashes on some Spaces hardware.
    demo.launch(ssr_mode=False)
