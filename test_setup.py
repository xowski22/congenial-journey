import torch
import pytorch3d
import open3d as o3d
import trimesh
import numpy as np
import matplotlib.pyplot as plt

print(f"PyTorch version: {torch.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

print(f"PyTorch3D version: {pytorch3d.__version__}")

vertices = torch.tensor([
    [0,0,0],
    [1,0,0],
    [0,1,0]
], dtype=torch.float32)

faces = torch.tensor([
    [0,1,2]
], dtype=torch.int64)

print("Successfully created a simple triangle mesh!")
