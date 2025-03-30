import os
import torch
import trimesh
import numpy as np
from torch.utils.data import DataLoader, Dataset

class MeshDataset(Dataset):

    def __init__(self, root_dir, category='chair', transform=None):
        self.root_dir = root_dir
        self.category = category
        self.transform = transform
        self.mesh_path = self._get_mesh_paths()

    def _get_mesh_paths(self):
        category_dir = os.path.join(self.root_dir, self.category, "train")
        mesh_paths = []

        for filename in os.listdir(category_dir):
            if filename.endswith(".off"):
                mesh_paths.append(os.path.join(category_dir, filename))

        return mesh_paths

    def __len__(self):
        return len(self.mesh_path)

    def __getitem__(self, idx):
        mesh_path = self.mesh_path[idx]
        mesh = trimesh.load_mesh(mesh_path)

        points = mesh.sample(2048)

        points_tensor = torch.tensor(points, dtype=torch.float32)

        if self.transform:
            points_tensor = self.transform(points_tensor)

        return points_tensor

def get_dataloader(config):
    dataset = MeshDataset(config.data_dir)
    dataloader = DataLoader(
            dataset,
            batch_size= config.batch_size,
            shuffle=True,
            num_workers=config.num_workers
    )
    return dataloader