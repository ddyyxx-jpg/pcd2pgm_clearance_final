#!/usr/bin/env python3
"""Step 3 - fuse the two surveys into one field-frame point cloud.

Inputs
  --ref-field : reference survey already expressed in field coordinates
                (x in [0, 28], y in [0, 16], z = height above field floor).
                This is the ERASOR2 cloud produced by the mapping front-end.
  --src       : second survey in its own frame (e.g. another team's venue scan).
  --matrix    : 4x4 matrix from step 1/2, mapping src -> venue frame.

The venue frame is converted to field coordinates with the fixed tournament
frame transform (defaults match the RMUC 2026 arena / SUMA map frame):

    field_xy = (venue_xy - center) @ R(-rot) + translate
    height   = venue_z - (a * field_x + b * field_y + c) - datum

`--datum auto` re-datums the source cloud so that its floor histogram peak
becomes 0, matching the reference cloud convention.

Usage:
  python 03_fuse_scans.py \
      --ref-field erasor2_field.pcd --src red1.pcd \
      --matrix out/red1_to_venue_refined.txt \
      --out out/fused_field.pcd
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pcd_io import load_pcd_xyz, write_pcd  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-field", required=True, help="reference survey in field coords")
    ap.add_argument("--src", required=True, help="second survey in its own/venue frame")
    ap.add_argument("--matrix", required=True, help="4x4 src -> venue matrix")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rot-deg", type=float, default=52.61, help="venue -> field rotation (deg)")
    ap.add_argument("--center", type=float, nargs=2, default=[8.09, -6.71], help="venue center (x y)")
    ap.add_argument("--translate", type=float, nargs=2, default=[14.006, 8.265], help="field origin offset")
    ap.add_argument("--plane", type=float, nargs=3, default=[-0.0013, -0.0011, 0.556],
                    help="ground plane z = a*x + b*y + c in field coords")
    ap.add_argument("--datum", default="auto", choices=["auto", "zero", "none"],
                    help="ground datum handling of the source cloud")
    ap.add_argument("--xmin", type=float, default=-0.5)
    ap.add_argument("--xmax", type=float, default=28.5)
    ap.add_argument("--ymin", type=float, default=-0.5)
    ap.add_argument("--ymax", type=float, default=16.5)
    ap.add_argument("--voxel", type=float, default=0.0, help="optional dedupe voxel (0 = off)")
    return ap.parse_args()


def venue_to_field(xyz, args):
    a = np.radians(args.rot_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    xy = (xyz[:, :2] - args.center) @ R.T + args.translate
    h = xyz[:, 2] - (args.plane[0] * xy[:, 0] + args.plane[1] * xy[:, 1] + args.plane[2])
    return np.column_stack([xy, h])


def auto_datum(h, bin_size=0.02, lo=-3.0, hi=6.0):
    hist, edges = np.histogram(h, bins=np.arange(lo, hi, bin_size))
    peak = edges[np.argmax(hist)] + bin_size / 2.0
    return peak


def main():
    args = parse_args()
    ref = load_pcd_xyz(args.ref_field)
    src = load_pcd_xyz(args.src)
    T = np.loadtxt(args.matrix)
    print("ref-field %d pts | src %d pts" % (len(ref), len(src)))

    src_v = (T[:3, :3] @ src.T).T + T[:3, 3]
    src_f = venue_to_field(src_v, args)

    datum = 0.0
    if args.datum == "auto":
        datum = auto_datum(src_f[:, 2])
        print("auto datum (floor peak) = %.3f m" % datum)
    elif args.datum == "zero":
        datum = 0.0
    src_f[:, 2] -= datum

    def crop(p):
        m = (
            (p[:, 0] >= args.xmin) & (p[:, 0] <= args.xmax)
            & (p[:, 1] >= args.ymin) & (p[:, 1] <= args.ymax)
        )
        return p[m]

    ref_c = crop(ref)
    src_c = crop(src_f)
    print("after field crop: ref=%d src=%d" % (len(ref_c), len(src_c)))

    fused = np.vstack([ref_c, src_c])
    if args.voxel > 0:
        key = np.floor(fused / args.voxel).astype(np.int64)
        key -= key.min(axis=0, keepdims=True)
        dims = key.max(axis=0) + 1
        lin = (key[:, 0] * dims[1] + key[:, 1]) * dims[2] + key[:, 2]
        _, keep = np.unique(lin, return_index=True)
        fused = fused[keep]
        print("voxel dedupe %g m -> %d pts" % (args.voxel, len(fused)))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_pcd(args.out, fused)
    print("fused field cloud -> %s (%d pts)" % (args.out, len(fused)))
    print("bbox x[%.2f, %.2f] y[%.2f, %.2f] z[%.2f, %.2f]"
          % (fused[:, 0].min(), fused[:, 0].max(), fused[:, 1].min(),
             fused[:, 1].max(), fused[:, 2].min(), fused[:, 2].max()))


if __name__ == "__main__":
    main()
