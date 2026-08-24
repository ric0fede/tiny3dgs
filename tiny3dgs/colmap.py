"""
Load a COLMAP sparse reconstruction (camera poses and point cloud) and
convert it into this codebase's Camera objects and point cloud tensors.

COLMAP's sparse/0/ folder contains:
  cameras.bin   intrinsics per camera (model, focal length, principal point)
  images.bin    pose (quaternion and translation) and keypoints for each image
  points3D.bin  sparse point cloud: XYZ and RGB per point

Coordinate convention: COLMAP stores the transform from world to camera as
X_cam = R*X_world + t, with the camera looking down +Z, x right, y DOWN, and
projects with u = fx*X/Z + cx, v = fy*Y/Z + cy. This codebase's Camera also
looks down +Z with x right, but treats y as UP and projects with
px = cx + fx*x/z, py = cy - fy*y/z. The two conventions match if COLMAP's
second rotation row is negated when building R_wc. Camera center: C = -R^T @ t.

Run COLMAP's image_undistorter first so both the images and the intrinsics
are plain PINHOLE. This codebase's renderer assumes an undistorted camera.
"""

import io
import os
import struct
import collections

import numpy as np
import torch

from .camera import Camera

CameraModel = collections.namedtuple("CameraModel", ["id", "name", "n_params"])
CAMERA_MODELS = {
    0: CameraModel(0, "SIMPLE_PINHOLE", 3),
    1: CameraModel(1, "PINHOLE", 4),
    2: CameraModel(2, "SIMPLE_RADIAL", 4),
    3: CameraModel(3, "RADIAL", 5),
    4: CameraModel(4, "OPENCV", 8),
    5: CameraModel(5, "OPENCV_FISHEYE", 8),
    6: CameraModel(6, "FULL_OPENCV", 12),
    7: CameraModel(7, "FOV", 5),
    8: CameraModel(8, "SIMPLE_RADIAL_FISHEYE", 4),
    9: CameraModel(9, "RADIAL_FISHEYE", 5),
    10: CameraModel(10, "THIN_PRISM_FISHEYE", 12),
}


def _read(fh, fmt):
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, fh.read(size))


def _open_buffered(path):
    """Read the whole file into memory before parsing, instead of many small reads."""
    with open(path, "rb") as fh:
        return io.BytesIO(fh.read())


def read_cameras_binary(path):
    cams = {}
    with _open_buffered(path) as fh:
        n = _read(fh, "<Q")[0]
        for _ in range(n):
            cam_id, model_id, width, height = _read(fh, "<iiQQ")
            model = CAMERA_MODELS[model_id]
            params = _read(fh, "<" + "d" * model.n_params)
            cams[cam_id] = dict(id=cam_id, model=model.name, width=width,
                                 height=height, params=np.array(params))
    return cams


def read_cameras_text(path):
    cams = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            cams[int(p[0])] = dict(id=int(p[0]), model=p[1], width=int(p[2]),
                                    height=int(p[3]),
                                    params=np.array([float(v) for v in p[4:]]))
    return cams


def read_images_binary(path):
    images = {}
    with _open_buffered(path) as fh:
        n = _read(fh, "<Q")[0]
        for _ in range(n):
            img_id, qw, qx, qy, qz, tx, ty, tz, cam_id = _read(fh, "<idddddddi")
            name = b""
            while True:
                c = fh.read(1)
                if c == b"\x00":
                    break
                name += c
            n_pts = _read(fh, "<Q")[0]
            fh.read(24 * n_pts)   # skip 2D keypoints + point3D ids, unused here
            images[img_id] = dict(id=img_id, qvec=np.array([qw, qx, qy, qz]),
                                   tvec=np.array([tx, ty, tz]),
                                   camera_id=cam_id, name=name.decode())
    return images


def read_images_text(path):
    images = {}
    with open(path) as fh:
        lines = [l for l in fh if not l.startswith("#") and l.strip()]
    for i in range(0, len(lines), 2):        # every second line is keypoints
        p = lines[i].split()
        images[int(p[0])] = dict(
            id=int(p[0]),
            qvec=np.array([float(v) for v in p[1:5]]),
            tvec=np.array([float(v) for v in p[5:8]]),
            camera_id=int(p[8]), name=p[9],
        )
    return images


def read_points3D_binary(path):
    xyz, rgb = [], []
    with _open_buffered(path) as fh:
        n = _read(fh, "<Q")[0]
        for _ in range(n):
            _pid, x, y, z, r, g, b, _err = _read(fh, "<QdddBBBd")
            track_len = _read(fh, "<Q")[0]
            fh.read(8 * track_len)
            xyz.append((x, y, z))
            rgb.append((r, g, b))
    return np.array(xyz, dtype=np.float32), np.array(rgb, dtype=np.float32) / 255.0


def read_points3D_text(path):
    xyz, rgb = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            xyz.append([float(v) for v in p[1:4]])
            rgb.append([float(v) for v in p[4:7]])
    return np.array(xyz, dtype=np.float32), np.array(rgb, dtype=np.float32) / 255.0


def qvec_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def intrinsics_from_params(cam):
    """Return (fx, fy, cx, cy). Models other than pinhole keep only their pinhole part."""
    p, model = cam["params"], cam["model"]
    if model == "SIMPLE_PINHOLE":
        return p[0], p[0], p[1], p[2]
    if model == "PINHOLE":
        return p[0], p[1], p[2], p[3]
    if model in ("SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
        return p[0], p[0], p[1], p[2]
    if model in ("OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE", "THIN_PRISM_FISHEYE"):
        return p[0], p[1], p[2], p[3]
    if model == "FOV":
        return p[0], p[1], p[2], p[3]
    raise ValueError(f"unsupported camera model: {model}")


def camera_from_colmap(img, cam, device="cpu", scale=1.0):
    """
    Build a Camera from a COLMAP image entry.

    scale: apply the same downscale factor used on the photos, so
    intrinsics stay consistent with the image resolution.
    """
    R = qvec_to_rotmat(img["qvec"])          # world -> camera, y down
    t = img["tvec"]
    C = -R.T @ t                              # camera center in world coords

    R_ours = R.copy()
    R_ours[1, :] *= -1.0                      # convert from COLMAP's y down to this codebase's y up

    fx, fy, cx, cy = intrinsics_from_params(cam)
    W = int(round(cam["width"] * scale))
    H = int(round(cam["height"] * scale))

    c = Camera.__new__(Camera)
    c.width, c.height, c.device = W, H, device
    c.R_wc = torch.tensor(R_ours, dtype=torch.float32, device=device)
    c.position = torch.tensor(C, dtype=torch.float32, device=device)
    c.fx = float(fx) * scale
    c.fy = float(fy) * scale
    c.cx = float(cx) * scale
    c.cy = float(cy) * scale
    return c


def load_colmap(sparse_dir, device="cpu", scale=1.0):
    """
    sparse_dir: folder holding cameras/images/points3D (.bin or .txt),
                typically <project>/sparse/0.

    Returns (cameras, image_names, points_xyz, points_rgb), sorted by
    filename so cameras[i] corresponds to image_names[i].
    """
    def pick(stem):
        b, t = os.path.join(sparse_dir, stem + ".bin"), os.path.join(sparse_dir, stem + ".txt")
        if os.path.exists(b):
            return b, "bin"
        if os.path.exists(t):
            return t, "txt"
        raise FileNotFoundError(f"neither {stem}.bin nor {stem}.txt in {sparse_dir}")

    cpath, cfmt = pick("cameras")
    ipath, ifmt = pick("images")
    ppath, pfmt = pick("points3D")

    cams_raw = read_cameras_binary(cpath) if cfmt == "bin" else read_cameras_text(cpath)
    imgs_raw = read_images_binary(ipath) if ifmt == "bin" else read_images_text(ipath)
    xyz, rgb = read_points3D_binary(ppath) if pfmt == "bin" else read_points3D_text(ppath)

    ordered = sorted(imgs_raw.values(), key=lambda d: d["name"])
    cameras = [camera_from_colmap(im, cams_raw[im["camera_id"]], device, scale)
               for im in ordered]
    names = [im["name"] for im in ordered]

    return (cameras, names,
            torch.tensor(xyz, dtype=torch.float32, device=device),
            torch.tensor(rgb, dtype=torch.float32, device=device))


def normalize_scene(points, cameras, target_radius=1.0):
    """
    Recenter on the point cloud's centroid and rescale to a fixed radius, so
    scenes of any original scale/origin land in the range the default
    learning rates and pruning thresholds are tuned for.
    """
    centre = points.mean(dim=0)
    pts = points - centre
    radius = pts.norm(dim=1).quantile(0.95).clamp_min(1e-6)
    factor = target_radius / radius

    pts = pts * factor
    for c in cameras:
        c.position = (c.position - centre) * factor
    return pts, float(factor), centre


def load_images(image_dir, names, size=None, device="cpu"):
    """
    Load photos as (H,W,3) float tensors in [0,1].

    `names` may be filenames relative to image_dir, or full paths (pass
    image_dir="" in that case).
    """
    from PIL import Image
    out = []
    for n in names:
        path = os.path.join(image_dir, n) if image_dir else n
        im = Image.open(path).convert("RGB")
        if size is not None and im.size != tuple(size):
            im = im.resize(size, Image.LANCZOS)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        out.append(torch.tensor(arr, device=device))
    return out
