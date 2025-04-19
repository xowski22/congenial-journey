import torch
import numpy as np
import gradio as gr
import sys
import os

sys.path.append('.')


class ModelInterface:
    def __init__(self, model_path, model_type="vae", device="cuda"):
        self.device = device
        self.model_type = model_type

        if model_type == "vae":
            self.model = PointCloudVAE(latent_dim=256, num_points=2048).to(device)
        elif model_type == "diffusion":
            self.model = PointCloudDiffusion(num_points=2048).to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()