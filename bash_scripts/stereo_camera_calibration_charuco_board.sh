#!/usr/bin/env bash

set -x

pydebug() {
    python -m debugpy --listen 5678 --wait-for-client "$@"
}

camera_name="stereo_microscope"

python -m cam_calib_lib.scripts.calibration.calibrate_stereo_camera_charuco  \
    --images_path datasets/microscope_dataset7/C2/data \
    --camera_name $camera_name\
    --squares_in_x_dim 10 \
    --squares_in_y_dim 7 \
    --square_size 0.0050 \
    --marker_size 0.0030 \
    --output_path output/mock_or_microscope_collection7_C2/ \
    --print_debug \
    --save_debug_images \
    --distortion_flags CALIB_FIX_K2 CALIB_FIX_K3 CALIB_ZERO_TANGENT_DIST \
    --stereo_flags CALIB_USE_INTRINSIC_GUESS \
    --exclude "0008 0027 0033 0034 0048 0051" 
    # --show_windows

# python -m cam_calib_lib.scripts.calibration.calibrate_stereo_camera_charuco  \
#     --images_path output/mock_or_microscope_collection1/collection5_02022026/sampled_frames \
#     --camera_name $camera_name\
#     --squares_in_x_dim 10 \
#     --squares_in_y_dim 7 \
#     --square_size 0.0050 \
#     --marker_size 0.0030 \
#     --output_path output/mock_or_microscope_collection1/collection5_02022026/calib_results2_$camera_name \
#     --print_debug \
#     --save_debug_images

# python -m cam_calib_lib.scripts.calibration.calibrate_stereo_camera_charuco  \
#     --images_path output/mock_or_microscope_collection1/collection4/frames_5fps \
#     --camera_name $camera_name\
#     --squares_in_x_dim 10 \
#     --squares_in_y_dim 7 \
#     --square_size 0.0050 \
#     --marker_size 0.0030 \
#     --output_path output/mock_or_microscope_collection1/collection4/calib_results_$camera_name \
#     --print_debug 
