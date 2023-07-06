import os
from typing import Any

from safetensors.torch import load_model, save_model

def save_models(model_dict: dict[str: Any], ckpt_dir: str):
    os.makedirs(ckpt_dir, exist_ok=True)
    for name, model in model_dict.items():
        save_model(model, os.path.join(ckpt_dir, f"{name}.safetensors"))


def load_models(model_dict: dict[str: Any], ckpt_dir: str):
    for name, model in model_dict.items():
        load_model(model, os.path.join(ckpt_dir, f"{name}.safetensors"))
