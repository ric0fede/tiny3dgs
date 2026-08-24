"""
Export the gaussian cloud to, and load it back from, a standard 3D Gaussian
Splatting PLY layout, so results can be opened in common splat viewers.

Format notes:
  * Values are stored without activations applied. Opacity is the raw
    logit and scales are raw log scales; viewers apply sigmoid() and exp()
    themselves. Writing values that are already activated is a common
    cause of a PLY rendering as garbage in other tools.
  * Rotation is a quaternion in (w, x, y, z) order, normalized.
  * SH coefficients are stored channel major: f_dc_0..2 are the degree 0
    term for R, G, B; f_rest holds the higher order terms as all of R's
    coefficients, then all of G's, then all of B's. At SH degree 3 (15
    higher order coefficients per channel) this is 45 f_rest_* fields;
    viewers infer the degree from that count.
  * nx, ny, nz are written as zeros. Gaussians have no meaningful normal,
    but the fields are part of the expected layout.
  * Binary little endian.
"""

import numpy as np
import torch


def _ply_header(n, extra_fields):
    lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
    ]
    lines += [f"property float {name}" for name in extra_fields]
    lines += ["end_header"]
    return ("\n".join(lines) + "\n").encode("ascii")


def field_names(n_rest):
    names = ["x", "y", "z", "nx", "ny", "nz"]
    names += [f"f_dc_{i}" for i in range(3)]
    names += [f"f_rest_{i}" for i in range(n_rest)]
    names += ["opacity"]
    names += [f"scale_{i}" for i in range(3)]
    names += [f"rot_{i}" for i in range(4)]
    return names


def export_ply(model, path, flip_y=False, flip_z=False):
    """
    Write `model` to `path` in the standard 3DGS PLY layout.

    flip_y / flip_z: negate that axis (position and the matching quaternion
    components) if a viewer's handedness disagrees with this codebase's.
    """
    with torch.no_grad():
        xyz = model.positions.detach().cpu().numpy().astype(np.float32).copy()

        sh = model.sh_coeffs.detach().cpu().numpy().astype(np.float32)
        f_dc = sh[:, :, 0]                       # (N,3)
        f_rest = sh[:, :, 1:].reshape(sh.shape[0], -1)  # (N,45)

        opacity = model.opacity_logits.detach().cpu().numpy().astype(np.float32).reshape(-1, 1)
        scales = model.log_scales.detach().cpu().numpy().astype(np.float32)

        q = model.rotations.detach().cpu()
        q = (q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)).numpy().astype(np.float32)

        if flip_y:
            xyz[:, 1] *= -1.0
            q[:, 0] *= -1.0   # w
            q[:, 2] *= -1.0   # y
        if flip_z:
            xyz[:, 2] *= -1.0
            q[:, 0] *= -1.0
            q[:, 3] *= -1.0

        normals = np.zeros_like(xyz)

        data = np.concatenate(
            [xyz, normals, f_dc, f_rest, opacity, scales, q], axis=1
        ).astype(np.float32)

    names = field_names(f_rest.shape[1])
    assert data.shape[1] == len(names), (data.shape, len(names))

    with open(path, "wb") as fh:
        fh.write(_ply_header(data.shape[0], names))
        fh.write(data.tobytes(order="C"))

    return path, data.shape


def load_ply(path, device="cpu"):
    """
    Read back a PLY written by export_ply, for example to check that a
    round trip through the format leaves the model unchanged, or to render
    it again with this codebase's own renderer. Only supports the layout
    written here (binary little endian, float32, SH degree 3).
    """
    from .gaussians import GaussianModel

    with open(path, "rb") as fh:
        header = b""
        while not header.endswith(b"end_header\n"):
            header += fh.readline()
        text = header.decode("ascii")
        n = int([l for l in text.splitlines() if l.startswith("element vertex")][0].split()[-1])
        props = [l.split()[-1] for l in text.splitlines() if l.startswith("property float")]
        raw = np.frombuffer(fh.read(n * len(props) * 4), dtype="<f4").reshape(n, len(props))

    idx = {name: i for i, name in enumerate(props)}
    n_rest = sum(1 for p in props if p.startswith("f_rest_"))

    model = GaussianModel(n_gaussians=n, device=device)
    with torch.no_grad():
        def col(name):
            return torch.tensor(raw[:, idx[name]].copy(), dtype=torch.float32, device=device)

        model.positions.data = torch.stack([col("x"), col("y"), col("z")], dim=1)
        model.opacity_logits.data = col("opacity").unsqueeze(-1)
        model.log_scales.data = torch.stack([col(f"scale_{i}") for i in range(3)], dim=1)
        model.rotations.data = torch.stack([col(f"rot_{i}") for i in range(4)], dim=1)

        dc = torch.stack([col(f"f_dc_{i}") for i in range(3)], dim=1)          # (N,3)
        rest = torch.stack([col(f"f_rest_{i}") for i in range(n_rest)], dim=1)  # (N,45)
        rest = rest.reshape(n, 3, n_rest // 3)
        model.sh_coeffs.data = torch.cat([dc.unsqueeze(-1), rest], dim=-1)

    return model
