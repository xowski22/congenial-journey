import matplotlib.pyplot as plt
import numpy as np
import torch
import open3d as o3d

def visualize_point_cloud(points, title="Point Cloud"):
    """Point cloud visualization."""
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd.estimate_normals()

    o3d.visualization.draw_geometries([pcd], window_name=title)

def visualize_batch(batch, max_samples=4):
    """Visualize a batch of point clouds."""

    batch = batch[:min(max_samples, batch.shape[0])]
    for i, points in enumerate(batch):
        visualize_point_cloud(points, title=f"Batch {i}")