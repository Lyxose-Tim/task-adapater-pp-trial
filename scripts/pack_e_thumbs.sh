#!/bin/bash
# Pack sampled JPEG thumbs for Stage-E ep1 queries (pretend class 78 + one other).
set -euo pipefail
FR=/root/rivermind-data/ta-pp/data/full/frames
OUT=/tmp/e_thumbs
rm -rf "$OUT"
mkdir -p "$OUT/70064" "$OUT/119959"
# q3 class 78
for f in 4 12 20 28 35 43 51 59; do
  cp "$FR/70064/img_$(printf '%05d' $f).jpg" "$OUT/70064/"
done
# q0 class 92
for f in 2 6 10 14 18 22 26 30; do
  cp "$FR/119959/img_$(printf '%05d' $f).jpg" "$OUT/119959/"
done
tar -C "$OUT" -czf /tmp/e_thumbs.tgz 70064 119959
ls -l /tmp/e_thumbs.tgz
echo PACK_OK
