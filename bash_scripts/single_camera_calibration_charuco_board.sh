#!/usr/bin/env bash

set -x

pydebug() {
    python -m debugpy --listen 5678 --wait-for-client "$@"
}

camera_name="right"
python -m cam_calib_lib.scripts.calibration.calibrate_single_cam_charuco \
    --images_path output/mock_or_microscope_collection1/charuco_collection1/data_image_only \
    --camera_name $camera_name \
    --squares_in_x_dim 10 \
    --squares_in_y_dim 7 \
    --square_size 0.0080 \
    --marker_size 0.0048 \
    --output_path output/mock_or_microscope_collection1/charuco_collection1/calib_results_$camera_name \
    --print_debug 