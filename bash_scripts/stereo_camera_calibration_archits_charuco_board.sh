#!/usr/bin/env bash
# Stereo calibration with the 20x15 DICT_4X4_250 ChArUco board.
#
# Data format expected by calibrate_stereo_camera_charuco (see load_stereo_images):
#   one flat folder with left-XXXX.png and right-XXXX.png, paired by frame number.

set -e
set -x

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Defaults to the repo's venv; override with PYTHON=/path/to/python ./script.sh
PYTHON=${PYTHON:-$repo_root/.venv/bin/python}

pydebug() {
    $PYTHON -m debugpy --listen 5678 --wait-for-client "$@"
}

camera_name="dvrk_classic"
dataset=$repo_root/charuco_dataset
# Results go in OpencvCameraCalibration/output, like the other scripts
output_path=$repo_root/output/charuco-test-dataset/results_with_full_dist_$camera_name

# The calibration script doesn't create the output folder itself
mkdir -p "$output_path"

# Board: 20 x 15 squares (long side = x). Marker/square ratio ~0.67.
# TODO: measure the printed board and set square_size / marker_size (metres).

# CALIB_ZERO_TANGENT_DIST
$PYTHON -m cam_calib_lib.scripts.calibration.calibrate_stereo_camera_charuco \
    --images_path "$dataset" \
    --camera_name $camera_name \
    --squares_in_x_dim 20 \
    --squares_in_y_dim 15 \
    --square_size 0.006 \
    --marker_size 0.004 \
    --aruco_dict DICT_4X4_250 \
    --output_path "$output_path" \
    --print_debug \
    --save_debug_images \
    --rect_line_spacing 50 \
    --distortion_flags CALIB_FIX_K3  \
    --stereo_flags CALIB_USE_INTRINSIC_GUESS
