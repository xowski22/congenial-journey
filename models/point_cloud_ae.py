import torch
import torch.nn.functional as F
import torch.nn as nn

class PointCloudEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, latent_dim)
        self.bn5 = nn.BatchNorm1d(256)