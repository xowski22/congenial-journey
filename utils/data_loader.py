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