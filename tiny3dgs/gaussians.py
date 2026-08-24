"""
GaussianModel: a set of learnable anisotropic 3D gaussians.

Each gaussian has:
  positions      (3,)   center
  covariance     (3,3)  built from a rotation quaternion and a scale per axis
  opacity        scalar in (0,1)
  sh_coeffs      (3,16) spherical harmonics color coefficients (degree 3)

Rotation and scale are optimized instead of the covariance directly, since a
covariance matrix would not stay positive semi definite under raw gradient
steps.
"""

import torch
import torch.nn as nn


def quat_to_rotmat(q):
    """Convert (N,4) unnormalized quaternions (w,x,y,z) to (N,3,3) rotation matrices."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(-1)
    R = torch.stack(
        [
            1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)
    return R


class GaussianModel(nn.Module):
    def __init__(self, n_gaussians, scene_radius=1.0, device="cpu", sh_degree=3):
        super().__init__()

        self.sh_degree = sh_degree

        positions = (torch.rand(n_gaussians, 3) * 2 - 1) * scene_radius
        log_scales = torch.log(torch.ones(n_gaussians, 3) * scene_radius * 0.10)
        rotations = torch.zeros(n_gaussians, 4)
        rotations[:, 0] = 1.0
        rotations = rotations + torch.randn(n_gaussians, 4) * 0.05
        opacity_logits = torch.zeros(n_gaussians, 1) - 2.0  # sigmoid(-2) ~= 0.12
        sh = torch.zeros(n_gaussians, 3, (sh_degree + 1) ** 2)
        sh[:, :, 0] = torch.rand(n_gaussians, 3) * 0.6 - 0.3

        self.positions = nn.Parameter(positions.to(device))
        self.log_scales = nn.Parameter(log_scales.to(device))
        self.rotations = nn.Parameter(rotations.to(device))
        self.opacity_logits = nn.Parameter(opacity_logits.to(device))
        self.sh_coeffs = nn.Parameter(sh.to(device))

    @property
    def scales(self):
        return torch.exp(self.log_scales)

    @property
    def opacities(self):
        return torch.sigmoid(self.opacity_logits)

    def covariance(self):
        """Build the (N,3,3) covariance in world space, from rotation and scale."""
        R = quat_to_rotmat(self.rotations)   # (N,3,3)
        S = self.scales                       # (N,3)
        M = R * S.unsqueeze(1)                # scale the columns of R by the semi-axes
        cov = M @ M.transpose(1, 2)           # R S S^T R^T
        return cov

    def n(self):
        return self.positions.shape[0]

    @classmethod
    def from_point_cloud(cls, points, colors, device="cpu"):
        """Initialize gaussians from a 3D point cloud and a color per point."""
        n = points.shape[0]
        model = cls(n_gaussians=n, scene_radius=1.0, device=device)

        with torch.no_grad():
            # Initial scale is the average distance to the nearest neighbors,
            # so gaussians roughly fill the gaps between points without
            # overlapping heavily or leaving holes. Computed in row blocks
            # to avoid the O(n^2) memory cost of a full distance matrix.
            k = min(3, max(1, n - 1))
            block = 2048
            knn_parts = []
            for i in range(0, n, block):
                d_blk = torch.cdist(points[i:i + block], points)
                rows = torch.arange(i, min(i + block, n), device=points.device)
                d_blk[torch.arange(rows.shape[0], device=points.device), rows] = float("inf")
                knn_parts.append(d_blk.topk(k, largest=False).values.mean(dim=1))
                del d_blk
            knn_dist = torch.cat(knn_parts, dim=0)
            init_scale = knn_dist.clamp_min(1e-3)

            model.positions.data = points.clone().to(device)
            model.log_scales.data = torch.log(init_scale).unsqueeze(-1).repeat(1, 3).to(device)

            SH_C0 = 0.28209479177387814
            model.sh_coeffs.data[:, :, 0] = ((colors - 0.5) / SH_C0).to(device)

        return model
