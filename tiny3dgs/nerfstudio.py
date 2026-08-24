"""
Loader for nerfstudio format datasets (transforms.json).

Coordinate convention (different from COLMAP): transforms.json stores a 4x4
matrix per frame that maps camera to world, with x right, y UP, z pointing
BACKWARD (the camera looks down -z). For this codebase's Camera (x right, y
up, +z forward), the camera axes in world coordinates are the columns of
that matrix: col0 = right, col1 = up, col2 = backward. So R_wc rows are
[col0, col1, -col2], and the position is the translation column.
"""

import json
import os

import numpy as np
import torch

from .camera import Camera


def load_transforms(path, device="cpu", scale=1.0):
    """
    path: either a transforms.json file or a folder containing one.

    Returns (cameras, image_paths). Intrinsics may be given per frame or globally;
    both are handled.
    """
    if os.path.isdir(path):
        root = path
        jpath = os.path.join(path, "transforms.json")
    else:
        root = os.path.dirname(path)
        jpath = path

    with open(jpath) as fh:
        meta = json.load(fh)

    frames = sorted(meta["frames"], key=lambda f: f["file_path"])
    cameras, paths = [], []

    for fr in frames:
        def g(key, default=None):
            return fr.get(key, meta.get(key, default))

        fx, fy = g("fl_x"), g("fl_y")
        cx, cy = g("cx"), g("cy")
        W, H = g("w"), g("h")
        if fx is None or W is None:
            raise ValueError("transforms.json is missing fl_x / w, this variant is not supported")
        if fy is None:
            fy = fx

        c2w = np.array(fr["transform_matrix"], dtype=np.float64)
        R_c2w = c2w[:3, :3]
        C = c2w[:3, 3]

        R_wc = np.stack([R_c2w[:, 0], R_c2w[:, 1], -R_c2w[:, 2]], axis=0)

        cam = Camera.__new__(Camera)
        cam.width = int(round(W * scale))
        cam.height = int(round(H * scale))
        cam.device = device
        cam.R_wc = torch.tensor(R_wc, dtype=torch.float32, device=device)
        cam.position = torch.tensor(C, dtype=torch.float32, device=device)
        cam.fx = float(fx) * scale
        cam.fy = float(fy) * scale
        cam.cx = float(cx) * scale
        cam.cy = float(cy) * scale
        cameras.append(cam)

        fp = fr["file_path"]
        if not os.path.isabs(fp):
            fp = os.path.join(root, fp)
        # Some datasets omit the file extension in file_path.
        if not os.path.exists(fp):
            for ext in (".png", ".jpg", ".JPG", ".jpeg"):
                if os.path.exists(fp + ext):
                    fp = fp + ext
                    break
        paths.append(fp)

    return cameras, paths


def read_ply_points(path, device="cpu"):
    """
    Read a plain XYZ+RGB point cloud PLY (binary little endian or ASCII).
    """
    with open(path, "rb") as fh:
        header = b""
        while not header.endswith(b"end_header\n"):
            chunk = fh.readline()
            if not chunk:
                raise ValueError("malformed PLY header")
            header += chunk
        text = header.decode("ascii", errors="ignore")
        n = int([l for l in text.splitlines() if l.startswith("element vertex")][0].split()[-1])
        ascii_fmt = "format ascii" in text

        props = []
        for line in text.splitlines():
            if line.startswith("property "):
                parts = line.split()
                props.append((parts[1], parts[2]))    # (dtype, name)

        if ascii_fmt:
            rows = []
            for _ in range(n):
                rows.append([float(v) for v in fh.readline().split()])
            arr = np.array(rows)
            names = [p[1] for p in props]
            xyz = arr[:, [names.index("x"), names.index("y"), names.index("z")]]
            if "red" in names:
                rgb = arr[:, [names.index("red"), names.index("green"), names.index("blue")]] / 255.0
            else:
                rgb = np.full_like(xyz, 0.5)
        else:
            np_dtype = {"float": "<f4", "float32": "<f4", "double": "<f8",
                         "uchar": "u1", "uint8": "u1", "int": "<i4", "uint": "<u4"}
            dt = np.dtype([(name, np_dtype[t]) for t, name in props])
            raw = np.frombuffer(fh.read(n * dt.itemsize), dtype=dt, count=n)
            xyz = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float32)
            if "red" in raw.dtype.names:
                rgb = np.stack([raw["red"], raw["green"], raw["blue"]], axis=1).astype(np.float32) / 255.0
            else:
                rgb = np.full_like(xyz, 0.5)

    return (torch.tensor(np.asarray(xyz, dtype=np.float32), device=device),
            torch.tensor(np.asarray(rgb, dtype=np.float32), device=device))


def random_points_from_cameras(cameras, n_points=50000, device="cpu", seed=0):
    """
    Fallback for datasets with poses but no point cloud: scatter points in
    the box spanned by the camera positions. Lower quality than a real
    structure from motion cloud, but lets optimization start.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    pos = torch.stack([c.position.cpu() for c in cameras], dim=0)
    centre = pos.mean(0)
    radius = (pos - centre).norm(dim=1).median()
    pts = (torch.rand(n_points, 3, generator=g) * 2 - 1) * radius * 0.6 + centre
    cols = torch.full((n_points, 3), 0.5)
    return pts.to(device), cols.to(device)
