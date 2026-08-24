#!/usr/bin/env python3
"""
Command line entry point: reconstruct a 3D Gaussian Splatting scene from a
COLMAP or nerfstudio dataset.

    python train.py /path/to/project --iters 3000 --res-scale 0.25
"""

import argparse

from tiny3dgs.reconstruct import reconstruct


def parse_args():
    p = argparse.ArgumentParser(
        description="Train a 3D Gaussian Splatting model from a COLMAP or nerfstudio dataset."
    )
    p.add_argument("project_dir",
                    help="Dataset folder (COLMAP sparse/0/ or nerfstudio transforms.json)")
    p.add_argument("--outdir", default=None,
                    help="Output folder for model.ply / comparison.png (default: project_dir)")
    p.add_argument("--device", default=None, help="cuda or cpu (default: detected automatically)")

    p.add_argument("--res-scale", type=float, default=0.25,
                    help="Downscale factor for the input photos")
    p.add_argument("--iters", type=int, default=3000,
                    help="Number of training iterations")
    p.add_argument("--lr-scale", type=float, default=1.0,
                    help="Global multiplier on the learning rates set per parameter")
    p.add_argument("--holdout-every", type=int, default=8,
                    help="Hold out every Nth view for evaluation (0 disables)")
    p.add_argument("--image-subdir", default=None,
                    help="Image folder to use, for example images_2 for an already downscaled set")
    p.add_argument("--max-points", type=int, default=None,
                    help="Cap on the initial point cloud size")
    p.add_argument("--targets-on-gpu", dest="targets_on_cpu", action="store_false",
                    help="Keep all target images on GPU instead of CPU")

    p.add_argument("--prune-every", type=int, default=500,
                    help="Iterations between pruning passes")
    p.add_argument("--max-scale-ratio", type=float, default=0.5,
                    help="Max gaussian scale as a fraction of scene extent")
    p.add_argument("--cap-scales", action="store_true",
                    help="Clamp gaussian scales hard after each optimizer step")

    p.add_argument("--densify-every", type=int, default=200,
                    help="Iterations between densification passes (0 disables)")
    p.add_argument("--densify-until", type=int, default=None,
                    help="Stop densifying after this iteration (default: 60% of iters)")
    p.add_argument("--densify-frac", type=float, default=0.05,
                    help="Fraction of gaussians densified per pass")
    p.add_argument("--grad-threshold", type=float, default=0.0002,
                    help="Gradient threshold used only when --densify-frac is unset")
    p.add_argument("--percent-dense", type=float, default=0.01,
                    help="Clone/split size boundary, as a fraction of scene extent")
    p.add_argument("--max-gaussians", type=int, default=600000,
                    help="Hard cap on gaussian count")

    p.add_argument("--sigma-cut", type=float, default=3.0,
                    help="Gaussian footprint radius, in standard deviations")
    p.add_argument("--tile-chunk", type=int, default=32,
                    help="Tiles processed per rasterizer batch")
    p.add_argument("--max-per-tile", type=int, default=None,
                    help="Cap on gaussians composited per tile")
    p.add_argument("--views-per-iter", type=int, default=1,
                    help="Training views rendered per optimizer step")
    p.add_argument("--log-every", type=int, default=100,
                    help="Iterations between log lines")

    return p.parse_args()


def main():
    args = parse_args()
    reconstruct(
        project_dir=args.project_dir,
        res_scale=args.res_scale,
        iters=args.iters,
        lr_scale=args.lr_scale,
        holdout_every=args.holdout_every,
        prune_every=args.prune_every,
        max_scale_ratio=args.max_scale_ratio,
        sigma_cut=args.sigma_cut,
        tile_chunk=args.tile_chunk,
        log_every=args.log_every,
        outdir=args.outdir,
        device=args.device,
        max_points=args.max_points,
        image_subdir=args.image_subdir,
        targets_on_cpu=args.targets_on_cpu,
        max_per_tile=args.max_per_tile,
        cap_scales=args.cap_scales,
        densify_every=args.densify_every,
        densify_until=args.densify_until,
        grad_threshold=args.grad_threshold,
        max_gaussians=args.max_gaussians,
        percent_dense=args.percent_dense,
        densify_frac=args.densify_frac,
        views_per_iter=args.views_per_iter,
    )


if __name__ == "__main__":
    main()
