#!/usr/bin/env python3
"""From pcd2pgm raw occupancy grid (built on the clearance-normalized cloud) to the final map.

Usage:
  python3 build_final_map.py --pcd data/rmuc_2026.pcd \
      --raw-pgm maps/raw_map.pgm --raw-yaml maps/raw_map.yaml \
      --out-pgm maps/rmuc_2026_clearance_final2.pgm --out-yaml maps/rmuc_2026_clearance_final2.yaml
"""
import argparse
import numpy as np
import yaml
from scipy import ndimage

RES = 0.05
OX, OY = 0.000891, 1.48e-05
W, H = 560, 321
HEXAGONS = [(6.55, 7.97), (21.40, 7.97)]   # midfield symmetric hexagon platforms (PCD frame)
HEX_R = 0.95
OUTSIDE = (0.52, 15.48)                    # field y-range (PCD frame)
BANDS = [
    ("north_B", 14.71, 15.31, 11.7, 16.6),
    ("north_A", 13.66, 14.28, 12.3, 16.6),
    ("south_B", 0.71, 1.21, 9.5, 15.6),
    ("south_A", 1.68, 2.40, 9.8, 15.4),
]


def read_pgm(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"P5"
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = (int(v) for v in line.split())
        f.readline()
        return np.frombuffer(f.read(), dtype=np.uint8, count=w * h).reshape(h, w).copy()


def load_pcd_xyz_normal_z(path):
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if line.strip() == b"DATA binary":
                break
        raw = f.read()
    return np.frombuffer(raw, dtype=np.float32, count=(len(raw) // 24) * 6).reshape(-1, 6)


def clean(x, min_blob, min_pocket):
    lab, n = ndimage.label(x, structure=np.ones((3, 3), int))
    size = ndimage.sum(x, lab, range(1, n + 1)) * RES * RES
    rm = np.zeros(n + 1, bool)
    for k in range(1, n + 1):
        if size[k - 1] < min_blob:
            rm[k] = True
    x = x & ~rm[lab]
    labf, nf = ndimage.label(~x, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    sf = ndimage.sum(~x, labf, range(1, nf + 1)) * RES * RES
    border_ids = set(labf[0, :]) | set(labf[-1, :]) | set(labf[:, 0]) | set(labf[:, -1])
    for k in range(1, nf + 1):
        if k not in border_ids and sf[k - 1] < min_pocket:
            x[labf == k] = True
    for b in range(2):
        x[b, :] = x[-1 - b, :] = True
        x[:, b] = x[:, -1 - b] = True
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcd", required=True)
    ap.add_argument("--raw-pgm", required=True)
    ap.add_argument("--raw-yaml", required=True)
    ap.add_argument("--out-pgm", required=True)
    ap.add_argument("--out-yaml", required=True)
    args = ap.parse_args()

    # ---- raw map -> full grid ----
    raw = read_pgm(args.raw_pgm) == 0
    meta = yaml.safe_load(open(args.raw_yaml))
    h, w = raw.shape
    c0 = int(round((meta["origin"][0] - OX) / RES))
    r0 = int(round((meta["origin"][1] - OY) / RES))
    full = np.full((H, W), 254, np.uint8)
    full[r0:r0 + h, c0:c0 + w] = np.where(raw, 0, 254)
    o = full == 0

    # ---- cleaning stage 1 ----
    o = clean(o, 0.5, 0.5)

    xs = OX + np.arange(W) * RES
    ys = OY + np.arange(H) * RES

    # ---- midfield hexagon platforms ----
    for (cx, cy) in HEXAGONS:
        X, Y = np.meshgrid(xs - cx, ys - cy)
        aa = HEX_R * np.sqrt(3) / 2
        o |= (np.abs(Y) <= HEX_R) & (np.abs(X) <= aa) & (np.abs(Y) + np.abs(X) / np.sqrt(3) <= HEX_R)

    # ---- outside field strips ----
    r_s = int(round((OUTSIDE[0] - OY) / RES))
    r_n = int(round((OUTSIDE[1] - OY) / RES))
    o[:r_s, :] = True
    o[r_n + 1:, :] = True

    # ---- fill small pockets ----
    labf, nf = ndimage.label(~o, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    sf = ndimage.sum(~o, labf, range(1, nf + 1)) * RES * RES
    for k in range(1, nf + 1):
        if sf[k - 1] < 1.5:
            o[labf == k] = True
    o = clean(o, 0.3, 1.2)

    # ---- per-cell low/high up-facing surfaces from the original cloud ----
    rec = load_pcd_xyz_normal_z(args.pcd)
    xyz, nz = rec[:, :3], rec[:, 5]
    bottom = (nz < -0.3) & (xyz[:, 2] < 0.05)
    keep = ~bottom
    upm = keep & (nz > 0.2)
    ix = np.clip(((xyz[:, 0] - OX) / RES).astype(int), 0, W - 1)
    iy = np.clip(((xyz[:, 1] - OY) / RES).astype(int), 0, H - 1)
    zlow = np.full((H, W), np.inf)
    np.minimum.at(zlow, (iy[upm], ix[upm]), xyz[upm, 2])
    has = np.isfinite(zlow)
    zhigh = np.full((H, W), -np.inf)
    np.maximum.at(zhigh, (iy[upm], ix[upm]), xyz[upm, 2])

    cells = {}
    for i in np.where(upm)[0]:
        cells.setdefault((iy[i], ix[i]), []).append(i)
    covered_sunken = np.zeros((H, W), bool)
    for (r, c), ids in cells.items():
        zs = xyz[np.array(ids), 2]
        zl = zlow[r, c]
        if 0.08 < zl < 0.20 and np.any((zs > zl + 0.10) & (zs <= zl + 0.50)):
            covered_sunken[r, c] = True
    sunken_floor = has & (zlow > 0.08) & (zlow < 0.20)
    grade_slab = has & (zhigh >= 0.25) & (zhigh <= 0.40)

    # ---- stage: corridor-band covered-sunken repair (free space on both sides) ----
    for (_, y0, y1, x0, x1) in BANDS:
        r0_, r1_ = int(round((y0 - OY) / RES)), int(round((y1 - OY) / RES))
        c0_, c1_ = int(round((x0 - OX) / RES)), int(round((x1 - OX) / RES))
        band = np.zeros((H, W), bool)
        band[r0_:r1_ + 1, c0_:c1_ + 1] = True
        r_center = (r0_ + r1_) // 2
        for r in range(max(0, r_center - 6), min(H, r_center + 7)):
            for c in range(max(0, c0_), min(W, c1_ + 1)):
                if not o[r, c] or not covered_sunken[r, c]:
                    continue
                left = np.any(~o[r, max(0, c - 20):c])
                right = np.any(~o[r, c + 1:c + 21])
                if left and right:
                    o[r, c] = False
    o = clean(o, 0.3, 1.2)

    # ---- stage: sunken-floor tunnel cells are passable ----
    for (_, y0, y1, x0, x1) in BANDS:
        r0_, r1_ = int(round((y0 - OY) / RES)), int(round((y1 - OY) / RES))
        c0_, c1_ = int(round((x0 - OX) / RES)), int(round((x1 - OX) / RES))
        band = np.zeros((H, W), bool)
        band[r0_:r1_ + 1, c0_:c1_ + 1] = True
        cut = o & band & sunken_floor
        o = o & ~cut
    o = clean(o, 0.3, 1.2)

    # ---- stage: grade-level slabs (tunnel roofs/beams at field level) are passable ----
    for (_, y0, y1, x0, x1) in BANDS:
        r0_, r1_ = int(round((y0 - OY) / RES)), int(round((y1 - OY) / RES))
        c0_, c1_ = int(round((x0 - OX) / RES)), int(round((x1 - OX) / RES))
        band = np.zeros((H, W), bool)
        band[r0_:r1_ + 1, c0_:c1_ + 1] = True
        cut = o & band & (sunken_floor | grade_slab)
        o = o & ~cut
    o = clean(o, 0.3, 1.2)

    # ---- write ----
    out = np.where(o, 0, 254).astype(np.uint8)
    hdr = ("# final map: clearance-ground normalization + corridor passable rules\n%d %d\n255\n" % (W, H)).encode()
    with open(args.out_pgm, "wb") as f:
        f.write(b"P5\n" + hdr + out.tobytes())
    with open(args.out_yaml, "w") as f:
        f.write("image: %s\nmode: trinary\nresolution: %s\n"
                "origin: [%s, %s, 0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n"
                % (args.out_pgm.split("/")[-1], RES, OX, OY))
    free = ~o
    lab, n = ndimage.label(free, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    sizes = np.sort(ndimage.sum(free, lab, range(1, n + 1)) * RES * RES)[::-1]
    print("occupancy %.2f%% | free components %d | largest free %.1f m2" % (100 * o.mean(), n, sizes[0]))


if __name__ == "__main__":
    main()
