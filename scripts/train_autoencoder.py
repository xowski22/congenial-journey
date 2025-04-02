import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
import sys
sys.path.append('.')

from configs.base_config import BaseConfig
from utils.data_utils import get_dataloader
from models.point_cloud_ae import PointCloudAutoencoder
from visualization.visualize import visualize_batch


def chamfer_distance(x, y):
    batch_size = x.shape[0]
    num_points_x = x.shape[1]
    num_points_y = y.shape[1]

    x = x.unsqueeze(2)
    y = y.unsqueeze(1)

    dist = torch.sum((x-y)**2, dim=-1)

    min_dist_x = torch.min(dist, dim=2)[0]
    min_dist_y = torch.min(dist, dim=1)[0]

    chamfer_dist = torch.mean(min_dist_x, dim=1) + torch.mean(min_dist_y, dim=1)

    return torch.mean(chamfer_dist)

def train(config):
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)

    model = PointCloudAutoencoder(config)
    model = model.to(config.device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    dataloader = get_dataloader(config)
    writer = SummaryWriter(log_dir=config.log_dir)

    global_step = 0

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")
        for batch in progress_bar:
            batch =batch.to(config.device)

            reconstruction, latent = model(batch)
            loss = chamfer_distance(batch, reconstruction)

