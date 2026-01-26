#! /usr/bin/env bash

python -m cam_calib_lib.scripts.regular_chessboard_calib.calibrate_single_camera \
    --images_path ./output/calib_011225/calibration_data/ \
    --camera left \
    --chessboard_rows 5 \
    --chessboard_cols 8 \
    --square_size 0.025 \
    --output_path ./output/calib_011225/calib_results_left