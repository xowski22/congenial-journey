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

def vae_loss(reconstruction, original, mu, logvar, chamfer_weight=1.0):
    chamfer = chamfer_distance(original, reconstruction)

    kl_loss = -0.5 + torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = chamfer_weight * chamfer + kl_loss

    return total_loss, chamfer, kl_loss

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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

            writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{config.num_epochs} Loss: {avg_loss:.6f}")
        writer.add_scalar("train/loss", avg_loss, global_step)

        if (epoch+1) % 10 == 0:
            checkpoint_path = os.path.join(config.save_dir, f"model_epoch_{epoch+1}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, checkpoint_path)

            with torch.no_grad():
                model.eval()
                sample_batch = next(iter(dataloader)).to(config.device)
                reconstruction, _ = model(sample_batch)

                sample_batch = sample_batch.cpu()
                reconstruction = reconstruction.cpu()

                for i in range(min(4, sample_batch.shape[0])):
                    writer.add_mesh(f"original_{i}", vertices=sample_batch[i].usqueeze(0), global_step=epoch)
                    writer.add_mesh(f"reconstruction_{i}", vertices=reconstruction[i].squeeze(0), global_step=epoch)


if __name__ == "__main__":
    config = BaseConfig()
    train(config)