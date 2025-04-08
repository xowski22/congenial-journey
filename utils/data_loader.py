import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import open3d as o3d
import trimesh

class ShapeNetDataset(Dataset):
    def __init__(self, root_dir, categories=None, transform=None, npoints=2048):
        self.root_dir = root_dir
        self.categories = categories
        self.transform = transform
        self.npoints = npoints
        self.mesh_paths = self._collect_mesh_paths()

    def _collect_mesh_paths(self):
        mesh_paths = []

        return mesh_paths

    def __len__(self):
        return len(self.mesh_paths)

    def __getitem__(self, idx):
        mesh_path = self.mesh_paths[idx]
        mesh = trimesh.load(mesh_path)
        points = mesh.sample(self.npoints)

        points_tensor = torch.tensor(points, dtype=torch.float32)

        if self.transform:
            points_tensor = self.transform(points_tensor)

        return points_tensor