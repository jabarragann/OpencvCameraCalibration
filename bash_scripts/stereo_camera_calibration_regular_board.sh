#!/usr/bin/env bash

set -x

pydebug() {
    python -m debugpy --listen 5678 --wait-for-client "$@"
}

python -m cam_calib_lib.scripts.calibration.calibrate_stereo_camera \
    --images_path output/calib_dvrksi_020326/calibration_data \
    --corners_per_board_row 8 \
    --corners_per_board_column 6 \
    --square_size 0.004 \
    --output_path output/calib_dvrksi_020326/calib_results_stereo

# python -m cam_calib_lib.scripts.calibration.calibrate_stereo_camera \
#     --images_path output/calib_dvrk11_011225/calibration_data \
#     --corners_per_board_row 8 \
#     --corners_per_board_column 5 \
#     --square_size 0.01 \
#     --output_path output/calib_dvrk11_011225/calib_results_stereo


## External testing dataset 
# python -m cam_calib_lib.scripts.calibration.calibrate_stereo_camera \
#     --images_path output/external_dataset/imgs/frames \
#     --corners_per_board_row 11 \
#     --corners_per_board_column 7 \
#     --square_size 0.003 \
#     --output_path output/external_dataset/calib_results_stereo