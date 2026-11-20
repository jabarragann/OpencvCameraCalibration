#!/usr/bin/env bash
# Convert stereo calibration results into dvrk_video camera_info yamls.
#
# Usage: ./make_dvrk_calibration_yaml.sh <results_dir> [rig_name]
#   results_dir: output_path of stereo_camera_calibration_archits_charuco_board.sh
#   rig_name:    prefix for camera_name (default: jhu_dVRK)
#
# Writes <results_dir>/dvrk_handeye/{left,right}.yaml from raw_results/*.npy with
# R = identity and P = [K | 0] (unrectified, as used for hand-eye), then prints
# the stereo reprojection error.

set -e

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=${PYTHON:-$repo_root/.venv/bin/python}

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <results_dir> [rig_name]"
    exit 1
fi

results_dir="$(realpath "$1")"
rig=${2:-jhu_dVRK}

$PYTHON - "$results_dir" "$rig" <<'EOF'
import sys
from pathlib import Path

import numpy as np

results_dir, rig = Path(sys.argv[1]), sys.argv[2]
raw = results_dir / "raw_results"
out = results_dir / "dvrk_handeye"
out.mkdir(exist_ok=True)

metadata = np.load(raw / "metadata.npy", allow_pickle=True).item()
width, height = metadata["image_size"]


def fmt(values, cols):
    rows = [", ".join(f"{v:12.6f}" for v in values[i : i + cols]) for i in range(0, len(values), cols)]
    return "[" + ",\n          ".join(rows) + "]"


for side in ["left", "right"]:
    K = np.load(raw / f"{side}_camera_matrix.npy").reshape(3, 3)
    D = np.load(raw / f"{side}_dist_coeffs.npy").flatten()
    P = np.hstack([K, np.zeros((3, 1))])

    yaml_str = f"""image_width: {width}
image_height: {height}
camera_name: {rig}/{side}
camera_matrix:
  rows: 3
  cols: 3
  data: {fmt(K.flatten(), 3)}
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: {len(D)}
  data: {fmt(D, len(D))}
rectification_matrix:
  rows: 3
  cols: 3
  data: {fmt(np.eye(3).flatten(), 3)}
projection_matrix:
  rows: 3
  cols: 4
  data: {fmt(P.flatten(), 4)}
"""
    (out / f"{side}.yaml").write_text(yaml_str)
    print(f"Wrote {out / f'{side}.yaml'}")

print(f"Stereo reprojection error (RMS): {metadata['reprojection_error']:.4f} px")
EOF
