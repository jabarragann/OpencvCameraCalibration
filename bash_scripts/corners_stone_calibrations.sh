#!/usr/bin/env bash

set -x

pydebug() {
    python -m debugpy --listen 5678 --wait-for-client "$@"
}

program=calibrate_stereo_camera
#program=stereo_calib_refinement

root_path=./output/surpass_cornerstone/monday_light_off/
python -m cam_calib_lib.scripts.calibration.$program \
    --images_path $root_path/frames \
    --corners_per_board_row 8 \
    --corners_per_board_column 6 \
    --square_size 0.004 \
    --output_path $root_path/${program}01 \
    --distortion_flags CALIB_FIX_K3 CALIB_ZERO_TANGENT_DIST \
    --stereo_flags CALIB_FIX_INTRINSIC
