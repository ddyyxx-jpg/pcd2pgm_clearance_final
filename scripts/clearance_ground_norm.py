#!/usr/bin/env python3
"""Ground-reference estimation + per-cell height normalization for a multi-level field.

Idea
----
The field has several ground levels (main floor ~0.30 m, sunken tunnel corridors
~0.12 m, platforms ~0.45 m, central highland ~0.5 m).  A single fixed z threshold
cannot build a usable occupancy grid from such a cloud.

For every 2-D cell we estimate a local ground reference z_ground(x, y):

1. sort the up-facing points of the cell by z and split them into "layers"
   (a gap larger than MERGE starts a new layer);
2. starting from the lowest layer, take the first layer that has CLEAR free
   space above it (no other layer inside [hi + MERGE, hi + CLEAR]) as ground;
3. if the lowest layer is covered by a higher structure (e.g. the sunken floor
   of a tunnel below its roof), the ground reference is given to the first
   layer above that has enough clearance (the roof top / field surface);
4. cells whose ground reference is much higher than the neighbourhood median
   (> HI) and that are not covered by a downward-facing surface are replaced by
   the neighbourhood median (prevents low platform tops from becoming "ground");
   iterate until stable, then 3x3 median smoothing and nearest-value fill.

Then z_norm = z - z_ground(x, y): the multi-level ground is flattened to a
single reference, so a fixed-threshold grid builder can be reused unchanged.

Usage
-----
  python3 scripts/clearance_ground_norm.py --pcd data/rmuc_2026.pcd \
      --out maps/rmuc_2026_clearance_norm.pcd [--diag maps/ground_diag.png]
"""
import argparse
import numpy as np
from scipy import ndimage
from collections import defaultdict

# grid used for the RMUC2026 simulation cloud (0.05 m, 560 x 321)
RES = 0.05
OX, OY = 0.000891, 1.48e-05
W, H = 560, 321
CLEAR = 0.30       # free space required above a candidate ground layer [m]
MERGE = 0.05       # points closer than this belong to the same layer [m]
HI = 0.10          # tolerance against the neighbourhood median [m]


def load_pcd(path):
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if line.strip() == b"DATA binary":
                break
        raw = f.read()
    return np.frombuffer(raw, dtype=np.float32, count=(len(raw) // 24) * 6).reshape(-1, 6).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcd", required=True, help="input PCD (x y z normal_x normal_y normal_z)")
    ap.add_argument("--out", required=True, help="output normalized PCD")
    ap.add_argument("--diag", default="", help="optional diagnostic figure path")
    args = ap.parse_args()

    rec = load_pcd(args.pcd)
    xyz, nz = rec[:, :3], rec[:, 5]
    bottom = (nz < -0.3) & (xyz[:, 2] < 0.05)     # simulation bottom face -> drop
    keep = ~bottom
    up = keep & (nz > 0.2)
    ix = np.clip(((xyz[:, 0] - OX) / RES).astype(int), 0, W - 1)
    iy = np.clip(((xyz[:, 1] - OY) / RES).astype(int), 0, H - 1)

    cells = defaultdict(list)
    for i in np.where(up)[0]:
        cells[(iy[i], ix[i])].append(i)

    zlow = np.full((H, W), np.inf)
    np.minimum.at(zlow, (iy[up], ix[up]), xyz[up, 2])
    has_up = np.isfinite(zlow)
    dn = keep & (nz < -0.2)
    down_max = np.full((H, W), -np.inf)
    np.maximum.at(down_max, (iy[dn], ix[dn]), xyz[dn, 2])
    has_down = np.isfinite(down_max)

    G = np.full((H, W), np.nan)
    for (r, c), ids in cells.items():
        arr = np.array(ids)
        u = arr[nz[arr] > 0.2]
        if len(u) == 0:
            continue
        zs = np.sort(xyz[u, 2])
        layers = []
        start = prev = zs[0]
        for v in zs[1:]:
            if v - prev > MERGE:
                layers.append((start, prev))
                start = v
            prev = v
        layers.append((start, prev))
        for k, (lo, hi_) in enumerate(layers):
            covered = any((layers[j][0] > hi_ + MERGE) and (layers[j][0] <= hi_ + CLEAR)
                          for j in range(k + 1, len(layers)))
            if not covered:
                G[r, c] = (lo + hi_) / 2
                break

    Gwork = G.copy()
    it = 0
    for it in range(80):
        med = ndimage.generic_filter(Gwork, np.nanmedian, size=3, mode='nearest')
        bad = np.isnan(Gwork) | (Gwork > med + HI)
        if not bad.any():
            break
        Gwork[bad] = med[bad]
    if np.isnan(Gwork).any():
        ind = ndimage.distance_transform_edt(np.isnan(Gwork), return_distances=False, return_indices=True)
        Gwork = Gwork[tuple(ind)]
    Gwork = ndimage.median_filter(Gwork, size=3)
    print(f"ground reference: fallback iterations {it + 1}")

    zn = xyz[:, 2] - Gwork[iy, ix]
    floor_mask = up & ((np.abs(xyz[:, 2] - 0.30) < 0.04) | (np.abs(xyz[:, 2] - 0.13) < 0.04))
    fz = zn[floor_mask]
    print(f"ground points z_norm: median {np.median(fz):+.4f}  "
          f"|dz|<0.03 {100 * np.mean(np.abs(fz) < 0.03):.1f}%")
    rel = np.full((H, W), -9.0)
    np.maximum.at(rel, (iy[up], ix[up]), zn[up])
    kn = rel > -9
    print(f"up-facing relative height: P50 {np.percentile(rel[kn], 50):.3f}  "
          f"P90 {np.percentile(rel[kn], 90):.3f}  cells>0.05 {100 * np.mean(rel[kn] > 0.05):.1f}%")

    rec2 = rec.copy()
    rec2[:, 2] = zn.astype(np.float32)
    hdr = ("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n"
           "FIELDS x y z normal_x normal_y normal_z\nSIZE 4 4 4 4 4 4\nTYPE F F F F F F\n"
           "COUNT 1 1 1 1 1 1\nWIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA binary\n"
           % (len(rec2), len(rec2)))
    with open(args.out, "wb") as f:
        f.write(hdr.encode() + rec2.astype(np.float32).tobytes())
    print("saved", args.out)

    if args.diag:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(20, 8))
        im = ax[0].imshow(Gwork, origin='lower', extent=[0, 28, 0, 16.05],
                          cmap='turbo', interpolation='nearest')
        ax[0].set_title('clearance-based ground reference z_ground(x, y) [m]')
        plt.colorbar(im, ax=ax[0], fraction=0.03)
        ax[1].hist(zn[up], bins=np.arange(-0.5, 2.0, 0.02), color='tab:green')
        ax[1].set_yscale('log')
        ax[1].set_title('z_norm histogram (up-facing points)')
        plt.tight_layout()
        plt.savefig(args.diag, dpi=110)
        print("saved", args.diag)


if __name__ == "__main__":
    main()
