#!/bin/bash
# Run a grid of training runs, metrics only. Several copies may run in parallel: each claims a run
# by atomically creating runs/.claims/<name>, and skips claimed runs and runs that already exist.
#   scripts/sweep.sh wd      # weight decay {0, 0.3, 1, 3} x init seed {0, 1, 2}
#   scripts/sweep.sh frac    # training fraction 0.1..0.9 x init seed {0, 1, 2}, early stop at test loss 1e-6
# wd=1 / frac=0.3 with seed 0 is the main run, runs/seed0, and is skipped.
cd "$(dirname "$0")/.."
mkdir -p runs/.claims
case "$1" in
  wd)   values="0.3 3 0 1"; flag="--wd"; base="1";;
  frac) values="0.1 0.2 0.9 0.8 0.7 0.6 0.5 0.4 0.3"; flag="--frac"; base="0.3"; extra="--stop-test-loss 1e-6";;
  *)    echo "usage: $0 wd|frac"; exit 1;;
esac
for seed in 0 1 2; do
  for v in $values; do
    name="$1${v}_s${seed}"
    [ "$v" = "$base" ] && [ "$seed" = "0" ] && continue
    [ -e "runs/$name" ] && continue
    mkdir "runs/.claims/$name" 2>/dev/null || continue
    uv run python scripts/train.py $flag "$v" --init-seed "$seed" --save-every 0 $extra --name "$name" > "runs/$name.log" 2>&1
  done
done
