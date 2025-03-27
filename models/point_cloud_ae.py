import torch
import torch.nn.functional as F
import torch.nn as nn

class PointCloudEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(PointCloudEncoder, self).__init__()