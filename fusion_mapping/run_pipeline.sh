#!/usr/bin/env bash
# End-to-end fusion pipeline (edit the input paths first).
set -euo pipefail
cd "$(dirname "$0")"

REF=erasor2_venue.pcd          # reference survey (venue frame)
SRC=red1.pcd                   # second survey (own frame)
REF_FIELD=erasor2_field.pcd    # reference survey already in field frame
MODEL=official_field_model.pcd # official field model in field frame
OUT=out
mkdir -p "$OUT"

python3 scripts/01_register_venue.py \
  --ref "$REF" --src "$SRC" \
  --out-matrix "$OUT/src_to_venue.txt" --out-report "$OUT/reg_coarse.json"

python3 scripts/02_refine_point_to_plane.py \
  --ref "$REF" --src "$SRC" \
  --matrix "$OUT/src_to_venue.txt" --out-matrix "$OUT/src_to_venue_refined.txt" \
  --out-report "$OUT/reg_refined.json"

python3 scripts/03_fuse_scans.py \
  --ref-field "$REF_FIELD" --src "$SRC" --matrix "$OUT/src_to_venue_refined.txt" \
  --out "$OUT/fused_field.pcd"

python3 scripts/04_clean_dynamic.py \
  --cloud "$OUT/fused_field.pcd" --model "$MODEL" \
  --out "$OUT/fused_field_clean.pcd" --out-report "$OUT/clean.json"

python3 scripts/05_shape_by_official_model.py \
  --model "$MODEL" --scan "$OUT/fused_field_clean.pcd" \
  --out "$OUT/fused_official_shaped.pcd" --out-report "$OUT/shape.json"

python3 scripts/06_tunnel_repair.py \
  --shaped "$OUT/fused_official_shaped.pcd" --scan "$OUT/fused_field_clean.pcd" \
  --zones configs/tunnel_zones.json \
  --out "$OUT/fused_official_shaped_fixed.pcd"

echo "done -> $OUT"
