"""
Reconstruct a 3D Gaussian Splatting scene from real photographs, using a
COLMAP or nerfstudio reconstruction for camera poses and the initial point
cloud.

    from tiny3dgs.reconstruct import reconstruct
    model, history = reconstruct("/path/to/project", res_scale=0.25, iters=3000)

Expected COLMAP layout (after running image_undistorter):
    project/
      images/            undistorted photos
      sparse/0/          cameras.bin, images.bin, points3D.bin

Every Nth photo is held out of training so training loss alone can't hide
overfitting to the training views.
"""

import collections
import math
import os
import time

import numpy as np
import torch
from tqdm.auto import tqdm
from PIL import Image

from .colmap import load_colmap, load_images, normalize_scene
from .nerfstudio import load_transforms, read_ply_points, random_points_from_cameras
from .gaussians import GaussianModel
from .tiled import render_tiled
from .optim import make_optimizer, PositionLRDecay
from .density import prune, clamp_scales
from .densify import densify, DensificationStats
from .ply import export_ply


def _find_dataset(project_dir):
    """Detect whether project_dir is a COLMAP or nerfstudio dataset."""
    for sub in ("sparse/0", "sparse", "colmap/sparse/0"):
        cand = os.path.join(project_dir, *sub.split("/"))
        if os.path.exists(os.path.join(cand, "cameras.bin")) or \
           os.path.exists(os.path.join(cand, "cameras.txt")):
            return "colmap", cand
    if os.path.exists(os.path.join(project_dir, "transforms.json")):
        return "nerfstudio", os.path.join(project_dir, "transforms.json")
    raise FileNotFoundError(
        f"{project_dir}: no sparse/0 (COLMAP) or transforms.json (nerfstudio) found"
    )


def reconstruct(project_dir, res_scale=0.25, iters=3000, lr_scale=1.0,
                 holdout_every=8, prune_every=500, max_scale_ratio=0.5,
                 sigma_cut=3.0, tile_chunk=32, log_every=100,
                 outdir=None, device=None, max_points=None, image_subdir=None,
                 targets_on_cpu=True, max_per_tile=None, cap_scales=False,
                 densify_every=200, densify_until=None, grad_threshold=0.0002,
                 max_gaussians=600000, percent_dense=0.01,
                 densify_frac=0.05, views_per_iter=1):
    """
    Train a Gaussian Splatting model from a COLMAP or nerfstudio dataset.

    project_dir:    dataset folder (COLMAP sparse/0/ or nerfstudio transforms.json).
    res_scale:      downscale factor for the input photos.
    iters:          number of training iterations.
    lr_scale:       global multiplier on the learning rates set per parameter.
    holdout_every:  hold out 1 in N images for evaluation (0 disables).
    prune_every:    iterations between pruning passes.
    max_scale_ratio / cap_scales: max gaussian scale as a fraction of scene
                    extent, and whether to clamp it hard. Loose or off by
                    default, since unbounded scenes need large gaussians
                    for distant geometry.
    sigma_cut:      gaussian footprint radius, in standard deviations.
    tile_chunk:     tiles processed per rasterizer batch.
    max_per_tile:   cap on gaussians composited per tile.
    outdir:         where to write outputs (defaults to project_dir).
    device:         "cuda" or "cpu" (detected automatically if None).
    max_points:     cap on the initial point cloud size.
    image_subdir:   image folder to use, for example "images_2" for a set
                    that is already downscaled.
    targets_on_cpu: keep target images in CPU RAM, moving one to GPU per step.
    densify_every / densify_until: how often to densify, and when to stop.
    grad_threshold: gradient threshold used only when densify_frac is None.
    densify_frac:   fraction of gaussians densified per pass, selected by
                    highest gradient in screen space.
    percent_dense:  size boundary between "clone" and "split" during densification.
    max_gaussians:  hard cap on gaussian count.
    views_per_iter: number of training views rendered per optimizer step
                    (gradient accumulation; the loss is averaged over the
                    batch so the learning rates set per parameter stay
                    valid and unchanged regardless of batch size).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if outdir is None:
        outdir = project_dir

    _rast = lambda m, cam, **kw: render_tiled(
        m, cam, bg_color=(0.0, 0.0, 0.0), sigma_cut=sigma_cut,
        tile_chunk=tile_chunk, max_per_tile=max_per_tile, **kw)
    print(f"device: {device}")

    kind, loc = _find_dataset(project_dir)
    print(f"detected format: {kind}")

    if kind == "colmap":
        cameras, names, xyz, rgb = load_colmap(loc, device=device, scale=res_scale)
        # Prefer an already downscaled image folder (for example images_2)
        # if set, to avoid decoding full resolution photos only to shrink
        # them.
        image_dir = os.path.join(project_dir, image_subdir or "images")
        image_paths = [os.path.join(image_dir, n) for n in names]
        print(f"COLMAP: {len(cameras)} poses, {xyz.shape[0]} 3D points")
    else:
        cameras, image_paths = load_transforms(loc, device=device, scale=res_scale)
        print(f"nerfstudio: {len(cameras)} poses")
        xyz = rgb = None
        # Look for a point cloud alongside the transforms.
        for cand in ("colmap/sparse/0", "sparse/0"):
            cdir = os.path.join(project_dir, *cand.split("/"))
            if os.path.exists(os.path.join(cdir, "points3D.bin")) or \
               os.path.exists(os.path.join(cdir, "points3D.txt")):
                _, _, xyz, rgb = load_colmap(cdir, device=device, scale=1.0)
                print(f"  point cloud from {cand}: {xyz.shape[0]} points")
                break
        if xyz is None:
            for name in ("sparse_pc.ply", "points3D.ply", "point_cloud.ply"):
                pth = os.path.join(project_dir, name)
                if os.path.exists(pth):
                    xyz, rgb = read_ply_points(pth, device=device)
                    print(f"  point cloud from {name}: {xyz.shape[0]} points")
                    break
        if xyz is None:
            xyz, rgb = random_points_from_cameras(cameras, 50000, device=device)
            print("  WARNING: no point cloud found, using random init "
                  "(much lower quality than a real SfM point cloud)")

    if max_points is not None and xyz.shape[0] > max_points:
        sel = torch.randperm(xyz.shape[0], device=device)[:max_points]
        xyz, rgb = xyz[sel], rgb[sel]
        print(f"  point cloud reduced to {max_points} points")

    xyz, factor, centre = normalize_scene(xyz, cameras, target_radius=1.0)
    print(f"  scene normalized (factor {factor:.4f})")

    W, H = cameras[0].width, cameras[0].height
    print(f"  training resolution: {W}x{H}")
    # Keep photos in CPU RAM and move one to GPU per iteration. A capture
    # of 100 to 300 photos held permanently in VRAM would cost more memory
    # than the training step itself.
    tgt_device = "cpu" if targets_on_cpu else device
    targets = load_images("", image_paths, size=(W, H), device=tgt_device)
    print(f"  {len(targets)} images loaded onto "
          f"{'CPU (moved to GPU per iteration)' if targets_on_cpu else device}")

    idx_all = list(range(len(cameras)))
    held = set(idx_all[::holdout_every]) if holdout_every else set()
    train_idx = [i for i in idx_all if i not in held]
    print(f"  {len(train_idx)} training views, {len(held)} held out")
    if views_per_iter > 1:
        print(f"  views_per_iter={views_per_iter}: {views_per_iter} sequential renders "
              f"per step, gradient accumulated into a single backward/optim.step\n")
    else:
        print()

    model = GaussianModel.from_point_cloud(xyz, rgb, device=device)

    def _make_opt(m):
        return make_optimizer(m, scene_extent=1.0, lr_scale=lr_scale)

    optim = _make_opt(model)
    sched = PositionLRDecay(optim, optim.param_groups[0]["lr"],
                             final_ratio=0.01, max_steps=iters)
    stats = DensificationStats(model.n(), device=device)
    if densify_until is None:
        densify_until = int(iters * 0.6)   # stop growing late, let the model settle

    history = []
    recent = collections.deque(maxlen=200)   # smooths loss noise across views
    t0 = time.time()
    pbar = tqdm(range(1, iters + 1), desc="training", unit="it")
    for it in pbar:
        # Sample a small batch of views_per_iter training views, with replacement.
        batch_idx = [train_idx[k] for k in
                     torch.randint(0, len(train_idx), (views_per_iter,)).tolist()]
        aux_list = [({} if densify_every else None) for _ in batch_idx]

        optim.zero_grad()
        per_view_losses = []
        preds_numel = []
        for i, aux in zip(batch_idx, aux_list):
            pred = _rast(model, cameras[i], aux=aux)
            tgt = targets[i].to(device, non_blocking=True) if targets_on_cpu else targets[i]
            per_view_losses.append((pred - tgt).abs().mean())
            preds_numel.append(pred.numel())

        # Mean, not sum, so the gradient scale stays independent of batch size.
        loss = torch.stack(per_view_losses).mean()
        loss.backward()

        if densify_every:
            for aux, n_elem in zip(aux_list, preds_numel):
                m2d = aux.get("mean2d_raw", aux.get("mean2d"))
                g = None
                if m2d is not None:
                    g = getattr(m2d, "absgrad", None)
                    if g is None:
                        g = m2d.grad
                    if g is not None and g.dim() == 3:
                        g = g.squeeze(0)      # (1,N,2) -> (N,2)
                if g is None:
                    if it == 1:
                        tqdm.write("  WARNING: no gradient available in screen space; "
                                   "densification would select gaussians at random. "
                                   "Disabling densification for this run.")
                    densify_every = 0
                    break
                else:
                    # Rescale to undo the 1/(H*W*3) from loss.mean(), so
                    # grad_threshold sits on the scale it's tuned for.
                    stats.add(g, scale=float(n_elem))

        optim.step()
        sched.step()
        if cap_scales:
            clamp_scales(model, 1.0, max_scale_ratio)

        history.append(loss.item())
        recent.append(loss.item())
        avg = sum(recent) / len(recent)
        pbar.set_postfix(avg=f"{avg:.4f}", n=model.n())

        if it % log_every == 0 or it == 1:
            # Report the moving average. A single iteration's loss swings
            # a lot between easy and hard views and isn't informative alone.
            psnr = -10.0 * math.log10(max(avg ** 2, 1e-12))
            tqdm.write(f"iter {it:5d}/{iters}  L1(avg200) = {avg:.4f}  "
                       f"~PSNR {psnr:.1f} dB  N = {model.n()}  ({time.time()-t0:.0f}s)")
        if densify_every and it % densify_every == 0 and it <= densify_until:
            model, optim, n_add = densify(
                model, stats, _make_opt, scene_extent=1.0,
                grad_threshold=grad_threshold, percent_dense=percent_dense,
                max_gaussians=max_gaussians, verbose=False,
                select_frac=densify_frac)
            if n_add:
                tqdm.write(f"  iter {it}: densify +{n_add} -> {model.n()} gaussians")
                sched.opt = optim

        if prune_every and it % prune_every == 0 and it < iters:
            model, optim, n_rm = prune(model, _make_opt, 1.0,
                                        max_scale_ratio=max_scale_ratio, verbose=False)
            if n_rm:
                frac = n_rm / (model.n() + n_rm)
                tqdm.write(f"  iter {it}: prune -{n_rm} -> {model.n()}")
                if frac > 0.05:
                    tqdm.write(
                        f"    WARNING: removed {100*frac:.0f}% of gaussians in one pass. "
                        f"On unbounded scenes this usually means max_scale_ratio "
                        f"({max_scale_ratio}) is too tight and is deleting the background.")
                sched.opt = optim

    print(f"\ntraining done in {time.time()-t0:.0f}s, final train L1 = {history[-1]:.4f}")

    # Evaluate on views never trained on.
    if held:
        with torch.no_grad():
            errs = [float((_rast(model, cameras[i]) - targets[i].to(device)).abs().mean())
                    for i in sorted(held)]
        print(f"HELD OUT L1 = {np.mean(errs):.4f}  (train {history[-1]:.4f}), "
              f"a large gap means it memorized the training views")

    # Comparison, side by side, on a held out view (or a training view if none held).
    show = sorted(held)[0] if held else train_idx[0]
    with torch.no_grad():
        pred = _rast(model, cameras[show])
    pair = torch.cat([targets[show].cpu(), pred.cpu()], dim=1).numpy()
    Image.fromarray((pair * 255).clip(0, 255).astype(np.uint8)).save(
        os.path.join(outdir, "comparison.png"))

    ply_path = os.path.join(outdir, "model.ply")
    export_ply(model, ply_path)
    print(f"saved comparison.png and {ply_path} ({model.n()} gaussians)")

    return model, history
