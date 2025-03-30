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


