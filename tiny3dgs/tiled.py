"""
Rasterizer based on tiles, with frustum and tile culling.

A dense renderer evaluates every gaussian at every pixel: cost and memory
are O(N * H * W) regardless of how small the gaussians are. This module
instead:
   1. splits the screen into 16x16 tiles,
   2. for each gaussian, finds only the tiles its 2D footprint touches,
   3. sorts those (tile, gaussian) pairs by tile then depth,
   4. composites each tile using only its own short list.
Cost becomes O(N + total_overlap), where total_overlap is the number of
(tile, gaussian) pairs, which for small gaussians is far smaller than
N * n_tiles. Implemented in pure PyTorch (differentiable, no custom kernel).

Pair list construction (vectorized, no Python loop): each gaussian covers a
tile range [tx0..tx1] x [ty0..ty1]. Gaussian indices are expanded with
repeat_interleave(counts), each pair's position within its gaussian's block
is recovered from the exclusive cumsum, and that flat offset becomes (dx,
dy) via divmod. Sorting uses a single composite key
tile_id * N + depth_rank, which sorts by tile first and depth second in one
argsort. Pairs are then scattered into a padded (n_tiles, K) grid; padding
slots get alpha = 0 so they are inert during compositing.
"""

import torch
from torch.utils.checkpoint import checkpoint

from .sh import eval_sh

TILE = 16


def _build_pairs(mean2d, radii, H, W, depth_rank, device):
    """Return (tile_id, gauss_idx) for every tile a gaussian's bbox touches."""
    n_tx = (W + TILE - 1) // TILE
    n_ty = (H + TILE - 1) // TILE
    n_tiles = n_tx * n_ty

    x0 = ((mean2d[:, 0] - radii) / TILE).floor().clamp(0, n_tx - 1).long()
    x1 = ((mean2d[:, 0] + radii) / TILE).floor().clamp(0, n_tx - 1).long()
    y0 = ((mean2d[:, 1] - radii) / TILE).floor().clamp(0, n_ty - 1).long()
    y1 = ((mean2d[:, 1] + radii) / TILE).floor().clamp(0, n_ty - 1).long()

    # Cull gaussians whose bbox is entirely off screen.
    on_screen = (
        (mean2d[:, 0] + radii >= 0) & (mean2d[:, 0] - radii < W)
        & (mean2d[:, 1] + radii >= 0) & (mean2d[:, 1] - radii < H)
    )

    w = (x1 - x0 + 1).clamp_min(0) * on_screen
    h = (y1 - y0 + 1).clamp_min(0) * on_screen
    counts = w * h

    total = int(counts.sum().item())
    if total == 0:
        return None, None, n_tiles, n_tx, n_ty

    N = mean2d.shape[0]
    gauss_idx = torch.repeat_interleave(torch.arange(N, device=device), counts)

    starts = torch.cumsum(counts, 0) - counts
    offset = torch.arange(total, device=device) - starts[gauss_idx]

    w_g = w[gauss_idx].clamp_min(1)
    dx = offset % w_g
    dy = offset // w_g
    tile_x = x0[gauss_idx] + dx
    tile_y = y0[gauss_idx] + dy
    tile_id = tile_y * n_tx + tile_x

    key = tile_id * N + depth_rank[gauss_idx]
    order = torch.argsort(key)
    return tile_id[order], gauss_idx[order], n_tiles, n_tx, n_ty


def render_tiled(gaussians, camera, low_pass=0.3, bg_color=(0.05, 0.05, 0.08),
                  sigma_cut=3.0, tile_chunk=None, max_per_tile=None,
                  depth_slice=256, early_stop=1e-3, aux=None,
                  near_plane=0.01, radius_clip_ratio=1.5):
    """
    Render one view. Same signature and semantics as a dense renderer that works pixel by pixel.

    sigma_cut:    footprint radius in standard deviations. 3.0 keeps ~99% of
                  each gaussian's mass; lowering it culls more aggressively.
    tile_chunk:   how many tiles to composite at once. None = all. Lower it
                  if you hit OOM at high resolution.
    max_per_tile: hard cap on gaussians kept per tile (keeps the frontmost
                  ones). None = no cap.
    depth_slice:  process each tile's sorted list in slices of this many
                  gaussians, so compositing can stop early. None = one shot.
    early_stop:   bail out of a bucket once every pixel's remaining
                  transmittance is below this. None disables.
    """
    device = gaussians.positions.device
    H, W = camera.height, camera.width

    p_cam = camera.world_to_camera(gaussians.positions)
    z = p_cam[:, 2]

    # Cull near the projection plane. The projection Jacobian scales with
    # 1/z, so a gaussian close to z=0 projects to a covariance orders of
    # magnitude too large and washes the render out into a flat color.
    # This matters most for indoor or 360 captures where the camera sits
    # inside the scene.
    valid = z > near_plane
    z = z.clamp_min(near_plane)

    cov_cam = camera.rotate_to_camera(gaussians.covariance())

    x, y = p_cam[:, 0], p_cam[:, 1]
    px = camera.cx + camera.fx * x / z
    py = camera.cy - camera.fy * y / z
    mean2d = torch.stack([px, py], dim=-1)
    if aux is not None:
        # Densification needs the gradient of the mean in screen space: it
        # measures how hard a gaussian is being pulled to cover pixels it
        # cannot reach, which is the signal used to decide where to split
        # or clone.
        mean2d.retain_grad()
        aux["mean2d"] = mean2d

    N = p_cam.shape[0]
    J = torch.zeros(N, 2, 3, device=device)
    J[:, 0, 0] = camera.fx / z
    J[:, 0, 2] = -camera.fx * x / (z * z)
    J[:, 1, 1] = -camera.fy / z
    J[:, 1, 2] = camera.fy * y / (z * z)

    cov2d = J @ cov_cam @ J.transpose(1, 2)
    cov2d = cov2d + low_pass * torch.eye(2, device=device)

    det = (cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] ** 2).clamp_min(1e-6)
    inv00 = cov2d[:, 1, 1] / det
    inv01 = -cov2d[:, 0, 1] / det
    inv11 = cov2d[:, 0, 0] / det

    # Footprint radius: sigma_cut * largest eigenvalue sqrt of cov2d.
    a, b, c = cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1]
    mid = 0.5 * (a + c)
    disc = (mid * mid - det).clamp_min(0).sqrt()
    lam_max = (mid + disc).clamp_min(1e-8)
    radii = sigma_cut * lam_max.sqrt()

    # Drop gaussians whose footprint is implausibly large for the image.
    if radius_clip_ratio is not None:
        max_r = radius_clip_ratio * max(H, W)
        valid = valid & (radii < max_r)

    # View direction is gaussian -> camera; flipping the sign inverts the
    # SH terms of odd degree (1 and 3) and breaks the color that depends on
    # viewing direction.
    view_dirs = gaussians.positions - camera.position.unsqueeze(0)
    view_dirs = view_dirs / view_dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    color = (eval_sh(gaussians.sh_coeffs, view_dirs) + 0.5).clamp(0.0, 1.0)
    opacity = gaussians.opacities.squeeze(-1) * valid.float()   # culled -> alpha 0

    depth_rank = torch.empty(N, dtype=torch.long, device=device)
    depth_rank[torch.argsort(z)] = torch.arange(N, device=device)

    tile_ids, gauss_idx, n_tiles, n_tx, n_ty_pad = _build_pairs(
        mean2d, radii * valid.float(), H, W, depth_rank, device
    )

    bg = torch.tensor(bg_color, device=device)
    if tile_ids is None:
        return bg.view(1, 1, 3).expand(H, W, 3).clone()

    # Pad pairs into a (n_tiles, K) grid.
    counts = torch.bincount(tile_ids, minlength=n_tiles)
    K_max = int(counts.max().item())
    if max_per_tile is not None:
        K_max = min(K_max, int(max_per_tile))

    tile_starts = torch.cumsum(counts, 0) - counts
    within = torch.arange(tile_ids.shape[0], device=device) - tile_starts[tile_ids]
    keep = within < K_max

    grid_idx = torch.zeros(n_tiles, K_max, dtype=torch.long, device=device)
    grid_mask = torch.zeros(n_tiles, K_max, dtype=torch.bool, device=device)
    flat = tile_ids[keep] * K_max + within[keep]
    grid_idx.view(-1)[flat] = gauss_idx[keep]
    grid_mask.view(-1)[flat] = True

    # Process tiles in buckets of similar occupancy, so a global K doesn't
    # force every tile to pay for the busiest tile in the image.
    order_tiles = torch.argsort(counts)
    ys_local = torch.arange(TILE, device=device, dtype=torch.float32) + 0.5
    xs_local = torch.arange(TILE, device=device, dtype=torch.float32) + 0.5
    yy, xx = torch.meshgrid(ys_local, xs_local, indexing="ij")   # (16,16)

    tile_index = torch.arange(n_tiles, device=device)
    tile_ox_all = (tile_index % n_tx) * TILE
    tile_oy_all = (tile_index // n_tx) * TILE

    step = n_tiles if tile_chunk is None else int(tile_chunk)
    canvas = torch.zeros(n_tiles, TILE, TILE, 3, device=device)

    def _slice_composite(gi, gm_f, ox, oy, T_in):
        """Composite one depth slice of a bucket, carrying transmittance in and out."""
        T = gi.shape[0]
        pxx = ox + xx.view(1, 1, TILE, TILE)
        pyy = oy + yy.view(1, 1, TILE, TILE)

        m2 = mean2d[gi]
        dx = pxx - m2[..., 0].unsqueeze(-1).unsqueeze(-1)
        dy = pyy - m2[..., 1].unsqueeze(-1).unsqueeze(-1)

        i00 = inv00[gi].unsqueeze(-1).unsqueeze(-1)
        i01 = inv01[gi].unsqueeze(-1).unsqueeze(-1)
        i11 = inv11[gi].unsqueeze(-1).unsqueeze(-1)

        power = -0.5 * (i00 * dx * dx + 2 * i01 * dx * dy + i11 * dy * dy)
        gw = torch.exp(power.clamp(max=0.0))
        alpha = (opacity[gi].unsqueeze(-1).unsqueeze(-1) * gw).clamp(0.0, 0.999)
        alpha = alpha * gm_f.unsqueeze(-1).unsqueeze(-1)

        one_minus = (1.0 - alpha).clamp_min(1e-6)
        cum = torch.cumprod(one_minus, dim=1)
        trans = torch.cat(
            [torch.ones(T, 1, TILE, TILE, device=device), cum[:, :-1]], dim=1
        )
        weight = (T_in.unsqueeze(1) * trans * alpha).unsqueeze(-1)
        col = color[gi].unsqueeze(-2).unsqueeze(-2)
        rgb = (weight * col).sum(dim=1)
        return rgb, T_in * cum[:, -1]

    # Checkpoint whenever gradients are needed, regardless of tile_chunk.
    # Without it every depth slice's activations stay alive until backward,
    # which is an easy OOM on a real scene.
    use_ckpt = torch.is_grad_enabled() and gaussians.positions.requires_grad

    for t0 in range(0, n_tiles, step):
        t1 = min(t0 + step, n_tiles)
        sel = order_tiles[t0:t1]
        T = sel.shape[0]
        K = min(int(counts[sel].max().item()), K_max)
        if K == 0:
            canvas = canvas.index_copy(0, sel, bg.view(1, 1, 1, 3).expand(T, TILE, TILE, 3))
            continue

        ox = tile_ox_all[sel].view(T, 1, 1, 1).float()
        oy = tile_oy_all[sel].view(T, 1, 1, 1).float()

        # Compositing in depth slices with early termination: gaussians in a
        # tile are sorted from front to back, so once every pixel's
        # remaining transmittance is near zero, everything behind
        # contributes about zero and can be skipped.
        slice_k = K if depth_slice is None else min(int(depth_slice), K)
        T_run = torch.ones(T, TILE, TILE, device=device)
        acc = torch.zeros(T, TILE, TILE, 3, device=device)

        for k0 in range(0, K, slice_k):
            k1 = min(k0 + slice_k, K)
            gi = grid_idx[sel, k0:k1]
            gm_f = grid_mask[sel, k0:k1].float()
            if use_ckpt:
                rgb, T_run = checkpoint(_slice_composite, gi, gm_f, ox, oy, T_run,
                                         use_reentrant=False)
            else:
                rgb, T_run = _slice_composite(gi, gm_f, ox, oy, T_run)
            acc = acc + rgb
            if early_stop is not None and float(T_run.detach().max()) < early_stop:
                break

        canvas = canvas.index_copy(0, sel, acc + T_run.unsqueeze(-1) * bg)

    # Scatter tiles back into the full image: (n_ty,n_tx,16,16,3) -> (H,W,3).
    Hp, Wp = n_ty_pad * TILE, n_tx * TILE
    image = (canvas.view(n_ty_pad, n_tx, TILE, TILE, 3)
                    .permute(0, 2, 1, 3, 4)
                    .reshape(Hp, Wp, 3))
    image = image[:H, :W]

    return image.clamp(0.0, 1.0)


def pair_stats(gaussians, camera, sigma_cut=3.0):
    """Diagnostic: number of (tile, gaussian) pairs vs. the dense O(N*H*W) cost."""
    device = gaussians.positions.device
    H, W = camera.height, camera.width
    p_cam = camera.world_to_camera(gaussians.positions)
    z = p_cam[:, 2].clamp_min(1e-4)
    cov_cam = camera.rotate_to_camera(gaussians.covariance())
    x, y = p_cam[:, 0], p_cam[:, 1]
    mean2d = torch.stack([camera.cx + camera.fx * x / z,
                           camera.cy - camera.fy * y / z], dim=-1)
    N = p_cam.shape[0]
    J = torch.zeros(N, 2, 3, device=device)
    J[:, 0, 0] = camera.fx / z; J[:, 0, 2] = -camera.fx * x / (z * z)
    J[:, 1, 1] = -camera.fy / z; J[:, 1, 2] = camera.fy * y / (z * z)
    cov2d = J @ cov_cam @ J.transpose(1, 2) + 0.3 * torch.eye(2, device=device)
    det = (cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] ** 2).clamp_min(1e-6)
    mid = 0.5 * (cov2d[:, 0, 0] + cov2d[:, 1, 1])
    lam = (mid + (mid * mid - det).clamp_min(0).sqrt()).clamp_min(1e-8)
    radii = sigma_cut * lam.sqrt()
    dr = torch.empty(N, dtype=torch.long, device=device)
    dr[torch.argsort(z)] = torch.arange(N, device=device)
    tids, _, n_tiles, _, _ = _build_pairs(mean2d, radii, H, W, dr, device)
    pairs = 0 if tids is None else tids.shape[0]
    dense_work = N * H * W
    tiled_work = pairs * TILE * TILE
    return dict(n_gaussians=N, n_tiles=n_tiles, pairs=pairs,
                dense_work=dense_work, tiled_work=tiled_work,
                speedup=dense_work / max(tiled_work, 1))
