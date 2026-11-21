#!/usr/bin/env python3
"""Step 5 - rebuild the field point cloud with the official model as shape reference.

Idea
  Survey data   -> dimensions / as-built pose   (never overridden by the model)
  Official model-> shape / topology             (never used as raw replacement)

Implementation
  1. estimate a smooth, piecewise-translation offset field between the official
     model and the cleaned fused scan:
       - split the field into 1.5 m x 1.5 m blocks,
       - per block take the median (dx, dy, dz) of matched model/scan points,
       - fill empty blocks by nearest valid block (distance transform) and
         smooth with a uniform filter -> continuous field, no block seams,
  2. move the official model points by that field,
  3. keep structure points whose neighbourhood is supported by the scan
     (`--evidence-dist`), keep the whole model floor flattened at z=0,
  4. voxel dedupe -> final regular, noise-free point cloud.

Usage:
  python 05_shape_by_official_model.py \
      --model official_field_model.pcd --scan out/fused_field_clean.pcd \
      --out out/fused_official_shaped.pcd
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import voxel_downsample  # noqa: E402
from pcd_io import load_pcd_xyz, write_pcd  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="official field model in field frame")
    ap.add_argument("--scan", required=True, help="cleaned fused scan in field frame")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-report", default=None)
    ap.add_argument("--xmin", type=float, default=-1.0)
    ap.add_argument("--xmax", type=float, default=29.0)
    ap.add_argument("--ymin", type=float, default=-1.0)
    ap.add_argument("--ymax", type=float, default=17.0)
    ap.add_argument("--block", type=float, default=1.5, help="offset field block size (m)")
    ap.add_argument("--match-dist", type=float, default=0.35, help="model<->scan match distance (m)")
    ap.add_argument("--min-pairs", type=int, default=25, help="minimum matches per block")
    ap.add_argument("--evidence-dist", type=float, default=0.35, help="structure evidence distance (m)")
    ap.add_argument("--floor-z", type=float, default=0.12, help="model points below this are floor")
    ap.add_argument("--dedupe", type=float, default=0.02, help="final voxel dedupe (m)")
    return ap.parse_args()


def fill_nan_smooth(field, sigma=1.0):
    """Fill NaN cells by nearest valid cell, then smooth."""
    valid = ~np.isnan(field)
    if valid.sum() == 0:
        return np.zeros_like(field)
    f = np.where(valid, field, 0.0)
    f = ndimage.median_filter(f, size=3)
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    f = f[idx[0], idx[1]]
    return ndimage.uniform_filter(f, size=3)


def main():
    args = parse_args()
    model = load_pcd_xyz(args.model)
    scan = load_pcd_xyz(args.scan)
    print("model %d pts | scan %d pts" % (len(model), len(scan)))

    scan_d = voxel_downsample(scan, 0.05)
    tree = cKDTree(scan_d)

    # ---- 1) per-block offset field -----------------------------------------
    nx = int((args.xmax - args.xmin) / args.block) + 2
    ny = int((args.ymax - args.ymin) / args.block) + 2
    OX = np.full((ny, nx), np.nan)
    OY = np.full((ny, nx), np.nan)
    OZ = np.full((ny, nx), np.nan)
    n_valid = 0
    for bi in range(ny):
        for bj in range(nx):
            bx = args.xmin + bj * args.block
            by = args.ymin + bi * args.block
            m = (
                (model[:, 0] >= bx) & (model[:, 0] < bx + args.block)
                & (model[:, 1] >= by) & (model[:, 1] < by + args.block)
            )
            if m.sum() < 40:
                continue
            mq = model[m]
            d, idx = tree.query(mq, k=1, workers=-1)
            g = d < args.match_dist
            if g.sum() < args.min_pairs:
                continue
            diff = scan_d[idx[g]] - mq[g]
            OX[bi, bj] = np.median(diff[:, 0])
            OY[bi, bj] = np.median(diff[:, 1])
            OZ[bi, bj] = np.median(diff[:, 2])
            n_valid += 1
    print("offset field: %d / %d blocks valid" % (n_valid, nx * ny))

    FX, FY, FZ = fill_nan_smooth(OX), fill_nan_smooth(OY), fill_nan_smooth(OZ)
    bi = np.clip(np.floor((model[:, 0] - args.xmin) / args.block).astype(int), 0, nx - 1)
    bj = np.clip(np.floor((model[:, 1] - args.ymin) / args.block).astype(int), 0, ny - 1)
    moved = model + np.column_stack([FX[bj, bi], FY[bj, bi], FZ[bj, bi]])

    # ---- 2) evidence gate for structures, floor kept and flattened ---------
    tree_scan = cKDTree(scan)
    d_scan, _ = tree_scan.query(moved, k=1, workers=-1)
    is_structure = moved[:, 2] > args.floor_z
    ev = moved[is_structure]
    d_ev, _ = tree_scan.query(ev, k=1, workers=-1) if len(ev) else (np.zeros(0), None)
    ev = ev[d_ev < args.evidence_dist]
    print("structure points: %d -> evidence-supported %d" % (is_structure.sum(), len(ev)))

    floor = moved[moved[:, 2] <= args.floor_z].copy()
    floor[:, 2] = 0.0
    print("floor points: %d (flattened to z=0)" % len(floor))

    out = np.vstack([ev, floor]) if len(ev) else floor
    if args.dedupe > 0:
        out = voxel_downsample(out, args.dedupe)
    print("shaped cloud: %d pts" % len(out))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_pcd(args.out, out)
    print("shaped cloud -> %s" % args.out)

    if args.out_report:
        med = np.median(d_scan)
        with open(args.out_report, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_points": int(len(model)),
                    "scan_points": int(len(scan)),
                    "offset_blocks_valid": int(n_valid),
                    "offset_field_std_m": [float(np.nanstd(OX)), float(np.nanstd(OY)), float(np.nanstd(OZ))],
                    "evidence_dist_m": args.evidence_dist,
                    "structure_points": int(len(ev)),
                    "floor_points": int(len(floor)),
                    "output_points": int(len(out)),
                    "output_to_scan_median_m": float(med),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )


if __name__ == "__main__":
    main()
