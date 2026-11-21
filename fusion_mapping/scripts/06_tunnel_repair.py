#!/usr/bin/env python3
"""Step 6 - local repair of tunnel / gate openings.

The official model is an idealised asset: openings that exist in the real venue
(tunnels, gates, passages) are sometimes modelled as solid geometry, or the
local fit in step 5 closes them. This step restores the openings by replacing
the shaped-model points inside manually specified zones with the real scan
points, so the passage stays empty while everything else keeps the clean model
shape.

Zones are given in a JSON file:
  [
    {"name": "south_tunnel", "x": [11.5, 18.0], "y": [0.5, 3.0]},
    {"name": "north_tunnel", "x": [11.5, 17.0], "y": [13.0, 15.8]}
  ]

Usage:
  python 06_tunnel_repair.py \
      --shaped out/fused_official_shaped.pcd --scan out/fused_field_clean.pcd \
      --zones configs/tunnel_zones.json --out out/fused_official_shaped_fixed.pcd
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import voxel_downsample  # noqa: E402
from pcd_io import load_pcd_xyz, write_pcd  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shaped", required=True, help="cloud from step 5 (official-model shaped)")
    ap.add_argument("--scan", required=True, help="cleaned fused scan with real openings")
    ap.add_argument("--zones", required=True, help="JSON list of tunnel zones")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dedupe", type=float, default=0.02)
    return ap.parse_args()


def in_zone(p, xr, yr):
    return (
        (p[:, 0] >= xr[0]) & (p[:, 0] <= xr[1])
        & (p[:, 1] >= yr[0]) & (p[:, 1] <= yr[1])
    )


def main():
    args = parse_args()
    shaped = load_pcd_xyz(args.shaped)
    scan = load_pcd_xyz(args.scan)
    with open(args.zones, "r", encoding="utf-8") as f:
        zones = json.load(f)
    print("shaped %d | scan %d | zones %d" % (len(shaped), len(scan), len(zones)))

    keep = np.ones(len(shaped), bool)
    added = []
    for z in zones:
        xr, yr = z["x"], z["y"]
        m_shaped = in_zone(shaped, xr, yr)
        m_scan = in_zone(scan, xr, yr)
        keep &= ~m_shaped
        added.append(scan[m_scan])
        print("  %-16s model pts removed %6d | scan pts inserted %7d"
              % (z.get("name", "zone"), m_shaped.sum(), m_scan.sum()))

    out = np.vstack([shaped[keep]] + added) if added else shaped[keep]
    if args.dedupe > 0:
        out = voxel_downsample(out, args.dedupe)
    print("repaired cloud: %d pts (was %d)" % (len(out), len(shaped)))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_pcd(args.out, out)
    print("repaired cloud -> %s" % args.out)


if __name__ == "__main__":
    main()
