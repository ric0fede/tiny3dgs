"""
Densification half of adaptive density control. Decides where the model
needs more gaussians.

A point cloud typically covers some regions of a scene densely and others
sparsely (periphery, distant background, grazing angles). A gaussian sitting
alone in a sparse region gets gradient pressure to explain more pixels than
it can cover, and since it cannot multiply, it stretches instead. That
produces long, spiky, washed out gaussians in regions that are not covered
well.

Signal: the norm of the gradient of a gaussian's mean in screen space,
averaged over recent iterations, measures exactly this frustration. A high
value means the gaussian is being pulled around because its region is not
reconstructed enough. Two remedies, chosen by the gaussian's size:

  CLONE (small, high gradient): the region needs more coverage at the same
         detail scale, so duplicate the gaussian, nudged along the gradient
         so the copy does not sit exactly on top of the original.

  SPLIT (large, high gradient): one gaussian is representing structure
         finer than itself, so replace it with several smaller ones, sampled
         from its own distribution so they land inside the volume it
         occupied.

Both are followed by pruning (see density.py) to keep the count bounded.
"""

import torch

from .gaussians import GaussianModel, quat_to_rotmat


class DensificationStats:
    """
    Accumulates the gradient signal in screen space between densify passes.

    The gradient's magnitude depends on how the photometric loss is
    reduced. If the loss is a mean over pixels (`.abs().mean()`), the
    gradient arriving here is scaled down by roughly 1/(H*W*3) relative to
    an unreduced (summed) loss, so a fixed threshold tuned for the summed
    case would never fire. `add()` takes a `scale` factor (typically the
    prediction's element count) so callers can undo that reduction and keep
    a stable threshold that does not depend on resolution.
    """

    def __init__(self, n, device="cpu"):
        self.grad_accum = torch.zeros(n, device=device)
        self.denom = torch.zeros(n, device=device)
        self.device = device

    def add(self, mean2d_grad, visible_mask=None, scale=1.0):
        if mean2d_grad is None:
            return
        g = mean2d_grad.norm(dim=-1) * scale
        if visible_mask is not None:
            g = g * visible_mask.float()
        n = min(g.shape[0], self.grad_accum.shape[0])
        self.grad_accum[:n] += g[:n]
        self.denom[:n] += 1.0

    def average(self):
        return self.grad_accum / self.denom.clamp_min(1.0)

    def reset(self, n):
        self.grad_accum = torch.zeros(n, device=self.device)
        self.denom = torch.zeros(n, device=self.device)


def _concat_model(model, parts, device):
    """Build a new GaussianModel from a list of parameter dicts."""
    total = sum(p["positions"].shape[0] for p in parts)
    new = GaussianModel(n_gaussians=total, device=device)
    with torch.no_grad():
        for key, attr in [("positions", new.positions), ("log_scales", new.log_scales),
                           ("rotations", new.rotations), ("opacity_logits", new.opacity_logits),
                           ("sh_coeffs", new.sh_coeffs)]:
            attr.data = torch.cat([p[key] for p in parts], dim=0).clone()
    return new


def densify(model, stats, optimizer_factory, scene_extent=1.0,
            grad_threshold=0.0002, percent_dense=0.01, split_n=2,
            max_gaussians=None, verbose=True, select_frac=0.05):
    """
    Returns (new_model, new_optimizer, n_added).

    select_frac: fraction of gaussians densified per pass, chosen by
                 highest gradient in screen space. Selection by percentile
                 is more robust than a fixed grad_threshold, since the
                 gradient's magnitude depends on the loss reduction,
                 resolution, and scene scale. A poorly set threshold can
                 select nothing at all, or nearly everything, and the
                 latter compounds into runaway gaussian counts over
                 several passes. Selecting a fixed top fraction each pass
                 bounds growth predictably instead.

    grad_threshold: used only when select_frac is None.
    percent_dense:  size boundary between "clone" and "split", as a
                    fraction of scene extent.
    max_gaussians:  hard cap so a run can't exhaust memory.
    """
    device = model.positions.device
    n = model.n()

    avg_grad = stats.average()[:n]
    big = model.scales.max(dim=1).values > percent_dense * scene_extent

    if select_frac is not None:
        k = max(1, int(n * float(select_frac)))
        idx = torch.argsort(avg_grad, descending=True)[:k]
        selected = torch.zeros(n, dtype=torch.bool, device=device)
        selected[idx] = True
    else:
        selected = avg_grad > grad_threshold

    to_clone = selected & (~big)
    to_split = selected & big

    if max_gaussians is not None:
        room = max_gaussians - n
        if room <= 0:
            return model, optimizer_factory(model), 0
        # If we would overshoot, keep only the candidates with the highest gradient.
        want = int(to_clone.sum()) + int(to_split.sum()) * (split_n - 1)
        if want > room:
            k = max(1, room // split_n)
            idx = torch.argsort(avg_grad, descending=True)[:k]
            keep = torch.zeros_like(selected)
            keep[idx] = True
            to_clone = to_clone & keep
            to_split = to_split & keep

    if int(to_clone.sum()) == 0 and int(to_split.sum()) == 0:
        return model, optimizer_factory(model), 0

    with torch.no_grad():
        base = dict(
            positions=model.positions.data,
            log_scales=model.log_scales.data,
            rotations=model.rotations.data,
            opacity_logits=model.opacity_logits.data,
            sh_coeffs=model.sh_coeffs.data,
        )
        keep_mask = ~to_split          # split replaces the original
        parts = [{k: v[keep_mask] for k, v in base.items()}]

        # clone
        if int(to_clone.sum()) > 0:
            c = {k: v[to_clone].clone() for k, v in base.items()}
            step = 0.01 * scene_extent
            jitter = torch.randn_like(c["positions"]) * step
            c["positions"] = c["positions"] + jitter
            parts.append(c)

        # split
        if int(to_split.sum()) > 0:
            s_idx = to_split.nonzero(as_tuple=True)[0]
            scales = model.scales[s_idx]                       # (M,3)
            R = quat_to_rotmat(model.rotations.data[s_idx])    # (M,3,3)
            for _ in range(split_n):
                unit = torch.randn(s_idx.shape[0], 3, device=device)
                offset = torch.einsum("bij,bj->bi", R, unit * scales)
                child = {k: v[s_idx].clone() for k, v in base.items()}
                child["positions"] = child["positions"] + offset
                child["log_scales"] = child["log_scales"] - torch.log(
                    torch.tensor(1.6 * split_n / 2.0, device=device))
                parts.append(child)

        new_model = _concat_model(model, parts, device)

    n_added = new_model.n() - n
    if verbose and n_added:
        print(f"  densify: +{n_added} (clone {int(to_clone.sum())}, "
              f"split {int(to_split.sum())}x{split_n}) -> {new_model.n()}")

    stats.reset(new_model.n())
    return new_model, optimizer_factory(new_model), n_added
