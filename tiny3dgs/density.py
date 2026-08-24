"""
Pruning half of adaptive density control.

Removes two kinds of gaussians that hurt both quality and speed if left in:

  Oversized gaussians. A few can grow until they span the whole scene,
  sitting at low opacity as a faint "ambient tint" that nudges the L1 loss
  down slightly without penalty. They add no real geometry, slow rendering,
  and dominate the view in a viewer as huge translucent discs.

  Dead gaussians. Opacity has collapsed toward zero. They cost memory and
  compute forever without drawing anything visible.

Pruning is on opacity and scale in world space; pruning by screen space
radius (used by some reference implementations) would need per view
bookkeeping this codebase does not keep.

Opacity reset (a separate trick) periodically clamps every opacity down,
forcing gaussians to earn their visibility again: gaussians that are
genuinely needed recover within a few hundred iterations, freeloaders stay
low and get pruned on the next pass.
"""

import torch

from .gaussians import GaussianModel


def prune_stats(model, scene_extent, max_scale_ratio=0.1, min_opacity=0.01):
    scales = model.scales.max(dim=1).values
    opacity = model.opacities.squeeze(-1)
    too_big = scales > max_scale_ratio * scene_extent
    too_faint = opacity < min_opacity
    return too_big, too_faint


def prune(model, optimizer_factory, scene_extent,
          max_scale_ratio=0.1, min_opacity=0.01, min_keep=32, verbose=True):
    """
    Returns (new_model, new_optimizer, n_removed).

    A pruned model is a new GaussianModel with fewer rows, so the optimizer
    is rebuilt via optimizer_factory(model). Adam's per parameter moment
    buffers are indexed by position and would be misaligned otherwise.
    """
    too_big, too_faint = prune_stats(model, scene_extent, max_scale_ratio, min_opacity)
    remove = too_big | too_faint
    keep = ~remove

    if keep.sum().item() < min_keep or remove.sum().item() == 0:
        return model, optimizer_factory(model), 0

    device = model.positions.device
    new_model = GaussianModel(n_gaussians=int(keep.sum()), device=device)
    with torch.no_grad():
        new_model.positions.data = model.positions.data[keep].clone()
        new_model.log_scales.data = model.log_scales.data[keep].clone()
        new_model.rotations.data = model.rotations.data[keep].clone()
        new_model.opacity_logits.data = model.opacity_logits.data[keep].clone()
        new_model.sh_coeffs.data = model.sh_coeffs.data[keep].clone()

    if verbose:
        print(f"  prune: -{remove.sum().item()} gaussians "
              f"({too_big.sum().item()} too large, {too_faint.sum().item()} too faint) "
              f"-> {new_model.n()} remaining")

    return new_model, optimizer_factory(new_model), int(remove.sum().item())


def clamp_scales(model, scene_extent, max_scale_ratio=0.1):
    """Cap gaussian scale in place, preventing oversized gaussians from forming."""
    with torch.no_grad():
        cap = torch.log(torch.tensor(max_scale_ratio * scene_extent,
                                      device=model.log_scales.device))
        model.log_scales.data.clamp_(max=float(cap))


def reset_opacity(model, value=0.05):
    """Force every opacity down; gaussians that matter recover, the rest get pruned."""
    with torch.no_grad():
        logit = torch.logit(torch.tensor(value, device=model.opacity_logits.device))
        model.opacity_logits.data.clamp_(max=float(logit))
