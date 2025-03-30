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

    def forward(self, x):
        x = x.transpose(2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.fc2(x)

        return x

class PointCloudDecoder(nn.Module):
    def __init__(self, latent_dim=256, num_points=2048):
        super().__init__()
        self.num_points = num_points

        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, num_points*3)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)

        x = x.view(-1, self.num_points, 3)
        return x

class PointCloudAutoencoder(nn.Module):
    def __init__(self, latent_dim=256, num_points=2048):
        super().__init__()
        self.encoder = PointCloudEncoder(latent_dim)
        self.decoder = PointCloudDecoder(latent_dim, num_points)

    def forward(self, x):
        lantent = self.encoder(x)
        reconstruction = self.decoder(lantent)
        return reconstruction, lantent

