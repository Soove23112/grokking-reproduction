#!/bin/bash
# Follow-up to the seed grid (docs/preregistration.md, "Follow-up: symmetry check and seeds 8–15").
# The symmetry run comes first. Same claim mechanism as scripts/seeds.sh.
cd "$(dirname "$0")/.."
mkdir -p runs/.claims
# name init_seed data_seed relabel
grid="sym_i0_d0_u7 0 0 7
seeds_i8_d0 8 0 1
seeds_i9_d0 9 0 1
seeds_i10_d0 10 0 1
seeds_i11_d0 11 0 1
seeds_i12_d0 12 0 1
seeds_i13_d0 13 0 1
seeds_i14_d0 14 0 1
seeds_i15_d0 15 0 1"
echo "$grid" | while read -r name init data u; do
  [ -e "runs/$name" ] && continue
  mkdir "runs/.claims/$name" 2>/dev/null || continue
  uv run python scripts/train.py --name "$name" --init-seed "$init" --data-seed "$data" --relabel "$u" \
    --epochs 20000 --save-every 500 > "runs/$name.log" 2>&1
done
