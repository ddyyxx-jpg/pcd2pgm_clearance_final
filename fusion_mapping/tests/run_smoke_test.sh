#!/usr/bin/env bash
# Fast end-to-end smoke test on synthetic data (no real dataset required).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=tests/out
rm -rf "$OUT"
mkdir -p "$OUT"

python3 tests/make_synthetic.py --out "$OUT"

python3 scripts/01_register_venue.py \
  --ref "$OUT/ref_venue.pcd" --src "$OUT/src_venue.pcd" \
  --out-matrix "$OUT/m1.txt" --out-report "$OUT/r1.json" \
  --zmin 0.5 --zmax 4.0 --res 0.3 --angle-step 6 --top-k 3 --ds 0.2 --iters 25

python3 scripts/02_refine_point_to_plane.py \
  --ref "$OUT/ref_venue.pcd" --src "$OUT/src_venue.pcd" \
  --matrix "$OUT/m1.txt" --out-matrix "$OUT/m2.txt" --out-report "$OUT/r2.json" \
  --zmin 0.5 --zmax 4.0 --ds 0.2

python3 scripts/03_fuse_scans.py \
  --ref-field "$OUT/ref_field.pcd" --src "$OUT/src_venue.pcd" --matrix "$OUT/m2.txt" \
  --out "$OUT/fused.pcd" --rot-deg 0 --center 0 0 --translate 0 0 --plane 0 0 0 --datum zero

python3 scripts/04_clean_dynamic.py \
  --cloud "$OUT/fused.pcd" --model "$OUT/official_model.pcd" \
  --out "$OUT/clean.pcd" --out-report "$OUT/clean.json" \
  --model-dist 0.3 --min-points 20 --dedupe 0

python3 scripts/05_shape_by_official_model.py \
  --model "$OUT/official_model.pcd" --scan "$OUT/clean.pcd" \
  --out "$OUT/shaped.pcd" --out-report "$OUT/shape.json" \
  --block 2.0 --min-pairs 10 --evidence-dist 0.5 --dedupe 0.05

python3 scripts/06_tunnel_repair.py \
  --shaped "$OUT/shaped.pcd" --scan "$OUT/clean.pcd" \
  --zones tests/tunnel_zones_test.json --out "$OUT/final.pcd" --dedupe 0.05

python3 - <<'PY'
import numpy as np, sys
sys.path.insert(0, "scripts")
from pcd_io import load_pcd_xyz
f = load_pcd_xyz("tests/out/final.pcd")
assert len(f) > 1000, "final cloud too small: %d" % len(f)
expected = np.loadtxt("tests/out/expected_pose.txt")
m1 = np.loadtxt("tests/out/m1.txt")
rot_err = abs(np.degrees(np.arctan2(m1[1, 0], m1[0, 0]) - np.arctan2(expected[1, 0], expected[0, 0])))
print("final cloud: %d points | registration rot error: %.2f deg" % (len(f), rot_err))
assert rot_err < 10.0, "registration rotation error too large: %.2f deg" % rot_err
print("SMOKE TEST PASSED")
PY
