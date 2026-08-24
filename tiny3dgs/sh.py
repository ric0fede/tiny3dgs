"""
Spherical harmonics color evaluation.

Each gaussian stores SH coefficients per color channel instead of a single
fixed RGB value. Evaluating them along the direction from the gaussian to
the camera gives color that depends on viewing direction (highlights, rim
light, etc.) without storing a separate texture per view.

Implements degree 3 (16 coefficients per channel): degree 0 (constant),
degree 1 (3 linear terms), degree 2 (5 quadratic terms), degree 3 (7 cubic
terms).
"""

import torch

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]


def eval_sh(sh_coeffs, view_dirs):
    """
    Evaluate color that depends on viewing direction, from per gaussian SH coefficients.

    sh_coeffs: (N, 3, 16) coefficients per gaussian, per color channel,
               ordered DC, then 3 degree 1, 5 degree 2, 7 degree 3 terms.
    view_dirs: (N, 3) unit vectors from gaussian to camera, in world space.

    Returns (N, 3) color centered at 0 (caller adds 0.5 and clamps to [0,1]
    so the color can be optimized with unconstrained gradients).
    """
    x = view_dirs[:, 0]
    y = view_dirs[:, 1]
    z = view_dirs[:, 2]

    dc = sh_coeffs[:, :, 0]
    c1 = sh_coeffs[:, :, 1]
    c2 = sh_coeffs[:, :, 2]
    c3 = sh_coeffs[:, :, 3]
    c4 = sh_coeffs[:, :, 4]
    c5 = sh_coeffs[:, :, 5]
    c6 = sh_coeffs[:, :, 6]
    c7 = sh_coeffs[:, :, 7]
    c8 = sh_coeffs[:, :, 8]
    c9 = sh_coeffs[:, :, 9]
    c10 = sh_coeffs[:, :, 10]
    c11 = sh_coeffs[:, :, 11]
    c12 = sh_coeffs[:, :, 12]
    c13 = sh_coeffs[:, :, 13]
    c14 = sh_coeffs[:, :, 14]
    c15 = sh_coeffs[:, :, 15]

    x = x.unsqueeze(-1)
    y = y.unsqueeze(-1)
    z = z.unsqueeze(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    color = SH_C0 * dc
    color = (
        color
        - SH_C1 * y * c1
        + SH_C1 * z * c2
        - SH_C1 * x * c3
    )
    color = (
        color
        + SH_C2[0] * xy * c4
        + SH_C2[1] * yz * c5
        + SH_C2[2] * (2.0 * zz - xx - yy) * c6
        + SH_C2[3] * xz * c7
        + SH_C2[4] * (xx - yy) * c8
    )
    color = (
        color
        + SH_C3[0] * y * (3 * xx - yy) * c9
        + SH_C3[1] * xy * z * c10
        + SH_C3[2] * y * (4 * zz - xx - yy) * c11
        + SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * c12
        + SH_C3[4] * x * (4 * zz - xx - yy) * c13
        + SH_C3[5] * z * (xx - yy) * c14
        + SH_C3[6] * x * (xx - 3 * yy) * c15
    )
    return color
