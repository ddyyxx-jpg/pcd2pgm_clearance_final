#!/usr/bin/env python3
"""Generate a small synthetic venue pair for a fast smoke test.

The scene is intentionally asymmetric (a tower, a few boxes, an octagonal ring
with one missing arc) so that venue registration has a unique solution.

Outputs (in --out dir): ref_venue.pcd, src_venue.pcd, ref_field.pcd,
official_model.pcd, expected_pose.txt
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from pcd_io import write_pcd  # noqa: E402


def build_scene(rng):
    """Asymmetric synthetic venue: floor + random buildings of different sizes.

    No ring / rectangle symmetry, so registration has a unique solution.
    """
    pts = []
    # floor
    n = 30000
    pts.append(np.column_stack([rng.uniform(0, 28, n), rng.uniform(0, 16, n), rng.normal(0, 0.01, n)]))

    # 12 rectangular "buildings": four vertical walls each, random footprint/height
    for _ in range(12):
        cx, cy = rng.uniform(3, 25), rng.uniform(3, 13)
        w, d = rng.uniform(1.5, 4.0), rng.uniform(1.5, 4.0)
        h = rng.uniform(1.0, 4.0)
        corners = [(cx - w / 2, cy - d / 2), (cx + w / 2, cy - d / 2),
                   (cx + w / 2, cy + d / 2), (cx - w / 2, cy + d / 2)]
        for k in range(4):
            x0, y0 = corners[k]
            x1, y1 = corners[(k + 1) % 4]
            for s in np.linspace(0, 1, 40):
                x, y = x0 + s * (x1 - x0), y0 + s * (y1 - y0)
                z = rng.uniform(0.05, h, 12)
                pts.append(np.column_stack([np.full(12, x) + rng.normal(0, 0.01, 12),
                                            np.full(12, y) + rng.normal(0, 0.01, 12), z]))
        # roof
        m = 600
        pts.append(np.column_stack([rng.uniform(cx - w / 2, cx + w / 2, m),
                                    rng.uniform(cy - d / 2, cy + d / 2, m),
                                    np.full(m, h) + rng.normal(0, 0.01, m)]))

    # 3 cylindrical towers at asymmetric positions
    for cx, cy, r, h in [(7.0, 5.0, 1.0, 3.5), (21.5, 11.5, 0.6, 2.2), (12.0, 13.0, 0.8, 2.8)]:
        th = rng.uniform(0, 2 * np.pi, 2500)
        rr = r * np.sqrt(rng.uniform(0, 1, 2500))
        pts.append(np.column_stack([cx + rr * np.cos(th), cy + rr * np.sin(th),
                                    rng.uniform(0, h, 2500)]))
    return np.vstack(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--rot-deg", type=float, default=25.0)
    ap.add_argument("--trans", type=float, nargs=3, default=[1.5, -0.8, 0.2])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    scene = build_scene(rng)
    write_pcd(os.path.join(args.out, "ref_venue.pcd"), scene)
    write_pcd(os.path.join(args.out, "ref_field.pcd"), scene)
    write_pcd(os.path.join(args.out, "official_model.pcd"), scene)

    a = np.radians(args.rot_deg)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    c = np.array([14.0, 8.0, 0.0])
    t = np.array(args.trans)
    src = (R @ (scene - c).T).T + c + t + rng.normal(0, 0.02, scene.shape)
    write_pcd(os.path.join(args.out, "src_venue.pcd"), src)

    expected = np.eye(4)
    # src = R @ (scene - c) + c + t   =>   scene = R^T @ (src - c - t) + c
    expected[:3, :3] = R.T
    expected[:3, 3] = c - R.T @ (c + t)
    np.savetxt(os.path.join(args.out, "expected_pose.txt"), expected, fmt="%.9f")
    print("synthetic scene: %d points -> %s" % (len(scene), args.out))
    print("expected src->ref rotation %.1f deg" % (-args.rot_deg))


if __name__ == "__main__":
    main()
