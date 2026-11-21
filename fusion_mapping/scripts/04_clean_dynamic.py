#!/usr/bin/env python3
"""Step 4 - remove dynamic objects / fog and flatten the ground.

The fused cloud still contains people, robots, smoke and ghost trails. With an
official field model available, the cleanest way to drop them is:

  1. model proximity: keep points within `--model-dist` of the official model
     surface (people / robots are not part of the field),
  2. connectivity: voxelise at 0.1 m, keep 26-connected components with at
     least `--min-points` points (fog is made of small isolated blobs),
  3. ground flatten: robust plane fit on near-floor points, subtract it.

The official model is also used as the shape reference in step 5; it must be
expressed in the same field frame (x in [0,28], y in [0,16], z height).

Usage:
  python 04_clean_dynamic.py \
      --cloud out/fused_field.pcd --model official_field_model.pcd \
      --out out/fused_field_clean.pcd --out-report out/clean_report.json
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import connected_component_filter, flatten_ground  # noqa: E402
from pcd_io import load_pcd_xyz, write_pcd  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", required=True, help="fused cloud in field frame")
    ap.add_argument("--model", required=True, help="official field model in field frame")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-report", default=None)
    ap.add_argument("--model-dist", type=float, default=0.30, help="keep points within this distance of model (m)")
    ap.add_argument("--voxel", type=float, default=0.1, help="voxel for connected components (m)")
    ap.add_argument("--min-points", type=int, default=100, help="minimum component size")
    ap.add_argument("--ground-band", type=float, default=0.35, help="band for ground plane fit (m)")
    ap.add_argument("--dedupe", type=float, default=0.03, help="final voxel dedupe (0 = off)")
    return ap.parse_args()


def main():
    args = parse_args()
    pts = load_pcd_xyz(args.cloud)
    model = load_pcd_xyz(args.model)
    print("input %d pts | model %d pts" % (len(pts), len(model)))

    tree = cKDTree(model)
    d, _ = tree.query(pts, k=1, workers=-1)
    pts2 = pts[d < args.model_dist]
    print("[model proximity %.2f m] %d -> %d (%.1f%%)"
          % (args.model_dist, len(pts), len(pts2), 100.0 * len(pts2) / len(pts)))

    pts3 = connected_component_filter(pts2, voxel=args.voxel, min_points=args.min_points)
    print("[connectivity] -> %d pts" % len(pts3))

    pts4, coef = flatten_ground(pts3, band=args.ground_band)
    ground_std = float(np.std(pts4[np.abs(pts4[:, 2]) < 0.12, 2])) if (np.abs(pts4[:, 2]) < 0.12).sum() > 10 else 0.0
    print("[ground flatten] plane z = %.4f x + %.4f y + %.4f | ground std %.3f m"
          % (coef[0], coef[1], coef[2], ground_std))

    out = pts4
    if args.dedupe > 0:
        key = np.floor(out / args.dedupe).astype(np.int64)
        key -= key.min(axis=0, keepdims=True)
        dims = key.max(axis=0) + 1
        lin = (key[:, 0] * dims[1] + key[:, 1]) * dims[2] + key[:, 2]
        _, keep = np.unique(lin, return_index=True)
        out = out[keep]
        print("[dedupe %g m] -> %d pts" % (args.dedupe, len(out)))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_pcd(args.out, out)
    print("clean cloud -> %s (%d pts)" % (args.out, len(out)))

    if args.out_report:
        with open(args.out_report, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "input_points": int(len(pts)),
                    "after_model_proximity": int(len(pts2)),
                    "after_connectivity": int(len(pts3)),
                    "after_dedupe": int(len(out)),
                    "model_dist_m": args.model_dist,
                    "voxel_m": args.voxel,
                    "min_points": args.min_points,
                    "ground_plane": [float(x) for x in coef],
                    "ground_std_m": ground_std,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )


if __name__ == "__main__":
    main()
