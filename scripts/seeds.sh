#!/bin/bash
# Key-frequency-across-seeds grid (docs/preregistration.md, "Key frequencies across seeds").
# 20 000 epochs, checkpoints every 500. Several copies may run in parallel: each claims a run
# by atomically creating runs/.claims/<name>, and skips claimed runs and runs that already exist.
cd "$(dirname "$0")/.."
mkdir -p runs/.claims
# name init_seed data_seed
grid="seeds_i1_d0 1 0
seeds_i2_d0 2 0
seeds_i3_d0 3 0
seeds_i4_d0 4 0
seeds_i5_d0 5 0
seeds_i6_d0 6 0
seeds_i7_d0 7 0
seeds_i0_d0_rep 0 0
seeds_i0_d1 0 1
seeds_i0_d2 0 2
seeds_i1_d1 1 1
seeds_i1_d2 1 2"
echo "$grid" | while read -r name init data; do
  [ -e "runs/$name" ] && continue
  mkdir "runs/.claims/$name" 2>/dev/null || continue
  uv run python scripts/train.py --name "$name" --init-seed "$init" --data-seed "$data" \
    --epochs 20000 --save-every 500 > "runs/$name.log" 2>&1
done
