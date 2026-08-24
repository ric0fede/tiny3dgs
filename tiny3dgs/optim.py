"""
Learning rates set per parameter, plus decay on the position rate.

The model's parameters live on very different scales and have very
different sensitivities: a step size reasonable for opacity logits is
catastrophic for positions, and an overly large learning rate on the SH
color terms that depend on viewing direction lets the optimizer cheat the
photometric loss by making color swing with view angle instead of fixing
geometry. That shows up as rainbow speckling that gets worse, not better,
as training loss goes down. Held out views, not training loss, are the
reliable signal.
"""

import torch


DEFAULTS = dict(
    position=0.00016,
    scaling=0.005,
    rotation=0.001,
    opacity=0.05,
    sh_dc=0.0025,
    sh_rest_divisor=20.0,   # higher order SH gets a much smaller lr
)


def make_optimizer(model, scene_extent=1.0, lr_scale=1.0, overrides=None):
    """
    scene_extent: rough scene radius. The position lr is scaled by it since
                  a "small step" means something different for a scene 1
                  unit across vs. one 50 units across.
    lr_scale:     global multiplier to speed up or slow down all rates at once.
    """
    cfg = dict(DEFAULTS)
    if overrides:
        cfg.update(overrides)

    groups = [
        {"params": [model.positions],
         "lr": cfg["position"] * scene_extent * lr_scale, "name": "position"},
        {"params": [model.log_scales],
         "lr": cfg["scaling"] * lr_scale, "name": "scaling"},
        {"params": [model.rotations],
         "lr": cfg["rotation"] * lr_scale, "name": "rotation"},
        {"params": [model.opacity_logits],
         "lr": cfg["opacity"] * lr_scale, "name": "opacity"},
        {"params": [model.sh_coeffs],
         "lr": cfg["sh_dc"] * lr_scale, "name": "sh"},
    ]
    return torch.optim.Adam(groups, eps=1e-15)


class PositionLRDecay:
    """
    Exponential decay of the position learning rate: gaussians move freely
    early on to find where they belong, then settle so late training
    refines appearance instead of rearranging geometry.
    """

    def __init__(self, optimizer, init_lr, final_ratio=0.01, max_steps=1000):
        self.opt = optimizer
        self.init_lr = init_lr
        self.final_lr = init_lr * final_ratio
        self.max_steps = max(1, max_steps)
        self.step_i = 0

    def step(self):
        self.step_i += 1
        t = min(self.step_i / self.max_steps, 1.0)
        lr = self.init_lr * (self.final_lr / self.init_lr) ** t
        for g in self.opt.param_groups:
            if g.get("name") == "position":
                g["lr"] = lr
        return lr
