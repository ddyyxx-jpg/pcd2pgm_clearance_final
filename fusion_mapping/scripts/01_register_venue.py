#!/usr/bin/env python3
"""Step 1 - whole-venue coarse registration of two point clouds.

Strategy (robust to the 8-fold rotational symmetry of the venue):
  1. take a structure band (default z in [2, 12] m) from both clouds,
  2. rasterise XY occupancy at 0.5 m,
  3. exhaustive 2D rotation search (0..360 deg) where for every angle the
     best translation is recovered by phase-correlation (FFT),
  4. keep the top-K candidates and refine each one with trimmed 3D
     point-to-point ICP (rigid, no scale),
  5. choose the candidate with the smallest median chamfer and write a 4x4
     matrix (source -> reference frame).

Usage:
  python 01_register_venue.py \
      --ref erasor2_venue.pcd --src red1.pcd \
      --out-matrix out/red1_to_venue.txt --out-report out/reg_report.json
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    icp_point_to_point,
    occupancy_2d,
    transform_points,
    voxel_downsample,
)
from pcd_io import load_pcd_xyz  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference (fixed) cloud, e.g. ERASOR2 venue map")
    ap.add_argument("--src", required=True, help="source cloud to be registered, e.g. second survey")
    ap.add_argument("--out-matrix", required=True, help="output 4x4 matrix txt (src -> ref)")
    ap.add_argument("--out-report", default=None, help="optional JSON report")
    ap.add_argument("--zmin", type=float, default=2.0, help="structure band lower z (m)")
    ap.add_argument("--zmax", type=float, default=12.0, help="structure band upper z (m)")
    ap.add_argument("--res", type=float, default=0.5, help="2D occupancy cell size (m)")
    ap.add_argument("--angle-step", type=float, default=2.0, help="rotation search step (deg)")
    ap.add_argument("--top-k", type=int, default=6, help="number of candidates refined with 3D ICP")
    ap.add_argument("--ds", type=float, default=0.25, help="voxel size for ICP input (m)")
    ap.add_argument("--trim", type=float, default=0.7, help="ICP trimming ratio")
    ap.add_argument("--iters", type=int, default=40, help="ICP iterations")
    return ap.parse_args()


def rotation2d(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def main():
    args = parse_args()
    ref = load_pcd_xyz(args.ref)
    src = load_pcd_xyz(args.src)
    print("[1/5] ref=%d src=%d points" % (len(ref), len(src)))

    ref_s = ref[(ref[:, 2] > args.zmin) & (ref[:, 2] < args.zmax)]
    src_s = src[(src[:, 2] > args.zmin) & (src[:, 2] < args.zmax)]
    print("      structure band z[%g,%g]: ref=%d src=%d" % (args.zmin, args.zmax, len(ref_s), len(src_s)))

    # ---- 2D rotation/translation search ------------------------------------
    A, (ax, ay) = occupancy_2d(ref_s, args.res)
    c_src = src_s[:, :2].mean(axis=0)
    candidates = []
    for ang in np.arange(0.0, 360.0, args.angle_step):
        R = rotation2d(ang)
        q2 = (src_s[:, :2] - c_src) @ R.T + c_src
        B, (bx, by) = occupancy_2d(np.column_stack([q2, src_s[:, 2]]), args.res)
        corr = fftconvolve(A, B[::-1, ::-1], mode="full")
        k = np.unravel_index(np.argmax(corr), corr.shape)
        score = corr.max() / (
            np.sqrt((A ** 2).sum()) * np.sqrt((B ** 2).sum()) + 1e-9
        )
        dx = ax + (k[1] - (B.shape[1] - 1)) * args.res - bx
        dy = ay + (k[0] - (B.shape[0] - 1)) * args.res - by
        candidates.append((float(score), float(ang), float(dx), float(dy)))
    candidates.sort(key=lambda c: -c[0])
    print("[2/5] top candidates (score, rot_deg, dx, dy):")
    for c in candidates[: args.top_k]:
        print("      %.3f  %6.1f  %7.2f  %7.2f" % c)

    # ---- 3D ICP refinement for each candidate -----------------------------
    ref_icp = voxel_downsample(ref_s, args.ds)
    src_icp = voxel_downsample(src_s, args.ds)
    tree = cKDTree(ref_icp)
    results = []
    for score, ang, dx, dy in candidates[: args.top_k]:
        # coarse search rotated src by R(ang) about c_src and then shifted it by
        # (dx, dy); the full src -> ref mapping is R(ang) (p - c) + c + d.
        R = rotation2d(ang)
        T = np.eye(4)
        T[:2, :2] = R
        T[:2, 3] = np.array([dx, dy]) + c_src - R @ c_src
        T, med = icp_point_to_point(src_icp, ref_icp, T, max_iter=args.iters, trim=args.trim)
        q = transform_points(T, src_icp)
        d, _ = tree.query(q, k=1, workers=-1)
        results.append((float(np.median(d)), float(np.percentile(d, 75)), float(ang), T))
    results.sort(key=lambda r: r[0])
    med, p75, ang, T = results[0]
    print("[3/5] best candidate: median chamfer %.4f m (P75 %.4f m, init rot %.1f deg)" % (med, p75, ang))

    # ---- report ------------------------------------------------------------
    scale = float(np.cbrt(np.linalg.det(T[:3, :3])))
    rot_deg = float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))
    print("[4/5] scale %.5f | rotation %.3f deg | translation (%.4f, %.4f, %.4f)"
          % (scale, rot_deg, T[0, 3], T[1, 3], T[2, 3]))

    os.makedirs(os.path.dirname(os.path.abspath(args.out_matrix)), exist_ok=True)
    np.savetxt(args.out_matrix, T, fmt="%.9f")
    print("[5/5] matrix written to %s" % args.out_matrix)

    if args.out_report:
        report = {
            "ref": args.ref,
            "src": args.src,
            "n_ref": int(len(ref)),
            "n_src": int(len(src)),
            "scale": scale,
            "rotation_deg": rot_deg,
            "translation": [float(T[0, 3]), float(T[1, 3]), float(T[2, 3])],
            "chamfer_median_m": med,
            "chamfer_p75_m": p75,
            "matrix": T.tolist(),
            "candidates": [
                {"score": c[0], "rot_deg": c[1], "dx": c[2], "dy": c[3]}
                for c in candidates[: args.top_k]
            ],
        }
        with open(args.out_report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print("      report written to %s" % args.out_report)


if __name__ == "__main__":
    main()
