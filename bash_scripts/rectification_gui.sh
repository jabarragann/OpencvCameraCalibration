#!/usr/bin/env bash

set -x

pydebug() {
    python -m debugpy --listen 5678 --wait-for-client "$@"
}

python src/cam_calib_lib/scripts/stereo_tools/rectify_image_gui.py \
 --l output/mock_or_microscope_collection1/collection5_02022026/sampled_frames/left_000200.png \
 --r output/mock_or_microscope_collection1/collection5_02022026/sampled_frames/right_000200.png \
 --c output/mock_or_microscope_collection1/collection5_02022026/calib_results_stereo_microscope/stereo.yaml 