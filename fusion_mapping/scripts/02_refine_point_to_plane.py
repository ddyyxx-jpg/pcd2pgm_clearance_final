#!/usr/bin/env python3
"""Step 2 - point-to-plane refinement of the venue matrix.

Point-to-point ICP is limited by the sampling of the two surveys. Running a
trimmed point-to-plane ICP on top of it (normals estimated from the reference
cloud) typically halves the median chamfer and removes the small rotational
residual left by step 1.

Usage:
  python 02_refine_point_to_plane.py \
      --ref erasor2_venue.pcd --src red1.pcd \
      --matrix out/red1_to_venue.txt --out-matrix out/red1_to_venue_refined.txt
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    estimate_normals,
    icp_point_to_plane,
    icp_point_to_point,
    transform_points,
    voxel_downsample,
)
from pcd_io import load_pcd_xyz  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--matrix", required=True, help="initial 4x4 matrix from step 1")
    ap.add_argument("--out-matrix", required=True)
    ap.add_argument("--out-report", default=None)
    ap.add_argument("--zmin", type=float, default=0.8)
    ap.add_argument("--zmax", type=float, default=20.0)
    ap.add_argument("--ds", type=float, default=0.15, help="voxel size for ICP (m)")
    ap.add_argument("--normal-k", type=int, default=16)
    return ap.parse_args()


def main():
    args = parse_args()
    ref = load_pcd_xyz(args.ref)
    src = load_pcd_xyz(args.src)
    T = np.loadtxt(args.matrix)

    ref_s = ref[(ref[:, 2] > args.zmin) & (ref[:, 2] < args.zmax)]
    src_s = src[(src[:, 2] > args.zmin) & (src[:, 2] < args.zmax)]
    ref_d = voxel_downsample(ref_s, args.ds)
    src_d = voxel_downsample(src_s, args.ds)
    print("ref=%d src=%d | ICP clouds: ref=%d src=%d" % (len(ref), len(src), len(ref_d), len(src_d)))

    tree = cKDTree(ref_d)

    def score(Tcur):
        q = transform_points(Tcur, src_d)
        d, _ = tree.query(q, k=1, workers=-1)
        return float(np.median(d)), float(np.percentile(d, 75))

    med0, p750 = score(T)
    print("[before] median %.4f m | P75 %.4f m" % (med0, p750))

    T, med_p2p = icp_point_to_point(src_d, ref_d, T, max_iter=30, trim=0.7)
    med1, p751 = score(T)
    print("[p2p   ] median %.4f m | P75 %.4f m" % (med1, p751))

    normals = estimate_normals(ref_d, k=args.normal_k)
    T, med, iters = icp_point_to_plane(src_d, ref_d, normals, T, max_iter=80, trim=0.7)
    med2, p752 = score(T)
    print("[p2pl  ] median %.4f m | P75 %.4f m (%d iterations)" % (med2, p752, iters))

    np.savetxt(args.out_matrix, T, fmt="%.9f")
    print("refined matrix -> %s" % args.out_matrix)

    if args.out_report:
        with open(args.out_report, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "before": {"median_m": med0, "p75_m": p750},
                    "after_p2p": {"median_m": med1, "p75_m": p751},
                    "after_p2pl": {"median_m": med2, "p75_m": p752},
                    "rotation_deg": float(np.degrees(np.arctan2(T[1, 0], T[0, 0]))),
                    "translation": [float(T[0, 3]), float(T[1, 3]), float(T[2, 3])],
                    "matrix": T.tolist(),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print("report -> %s" % args.out_report)


if __name__ == "__main__":
    main()
