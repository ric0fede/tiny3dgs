"""Pinhole camera model, with rotation built by looking at a target point."""

import math
import torch


class Camera:
    """Pinhole camera: position, rotation from world to camera space, and intrinsics."""

    def __init__(self, position, target, up, fov_deg, width, height, device="cpu"):
        self.width = width
        self.height = height
        self.device = device

        position = torch.tensor(position, dtype=torch.float32, device=device)
        target = torch.tensor(target, dtype=torch.float32, device=device)
        up = torch.tensor(up, dtype=torch.float32, device=device)

        forward = target - position
        forward = forward / forward.norm()
        right = torch.cross(forward, up, dim=-1)
        right = right / right.norm()
        true_up = torch.cross(right, forward, dim=-1)

        # Rows are the camera's own axes, expressed in world coordinates.
        self.R_wc = torch.stack([right, true_up, forward], dim=0)
        self.position = position

        focal = 0.5 * height / math.tan(math.radians(fov_deg) / 2)
        self.fx = focal
        self.fy = focal
        self.cx = width / 2
        self.cy = height / 2

    def world_to_camera(self, points_world):
        """Transform points in world space into camera space."""
        return (points_world - self.position) @ self.R_wc.T

    def rotate_to_camera(self, mat_world):
        """Rotate a batch of 3x3 matrices (e.g. covariances) into camera space."""
        R = self.R_wc.unsqueeze(0)
        return R @ mat_world @ R.transpose(1, 2)


def orbit_cameras(n_views, radius, height_offset, target, fov_deg, width, height, device="cpu"):
    """Generate cameras evenly spaced on a circle around a target point."""
    cams = []
    for i in range(n_views):
        theta = 2 * math.pi * i / n_views
        pos = [radius * math.cos(theta), height_offset, radius * math.sin(theta)]
        cams.append(Camera(pos, target, [0, 1, 0], fov_deg, width, height, device))
    return cams
