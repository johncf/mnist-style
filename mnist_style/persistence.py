import os

from safetensors.torch import load_model, save_model
from torch.nn import Module


def save_models(model_dict: dict[str, Module], ckpt_dir: str):
    os.makedirs(ckpt_dir, exist_ok=True)
    for name, model in model_dict.items():
        save_model(model, os.path.join(ckpt_dir, f"{name}.safetensors"))


def load_models(model_dict: dict[str, Module], ckpt_dir: str):
    for name, model in model_dict.items():
        load_model(model, os.path.join(ckpt_dir, f"{name}.safetensors"))
