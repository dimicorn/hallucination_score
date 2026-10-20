#!/usr/bin/env bash
# Download the HSC tract 9697 / patch 72 source subset (SCARLET models, PSFs, variance
# maps) from SampsonML/HSC_galaxy_tools. The upstream repo has no license, so the file
# is fetched rather than redistributed here.
set -euo pipefail
dest="$(dirname "$0")/../data/hsc_9697_72_subset.pkl"
expected=4c9dc294c8466751aba2fd9f0038cbf868f17b76f3629ec7a80760c821e451cf
mkdir -p "$(dirname "$dest")"
curl -fL -o "$dest" https://raw.githubusercontent.com/SampsonML/HSC_galaxy_tools/main/HSC_tract_9697_patch_72_subset.pkl
actual=$(shasum -a 256 "$dest" | cut -d' ' -f1)
[ "$actual" = "$expected" ] || { echo "checksum mismatch: $actual" >&2; exit 1; }
echo "ok: $dest"
