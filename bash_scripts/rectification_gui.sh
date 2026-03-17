#!/usr/bin/env bash

set -x

pydebug() {
    python -m debugpy --listen 5678 --wait-for-client "$@"
}

root_dir=output/surpass_dvrksi/stereo_calib_dvrksi_2026_02_17
python src/cam_calib_lib/scripts/stereo_tools/rectify_image_gui.py \
 --l ${root_dir}/calibration_data/left-0037.png \
 --r ${root_dir}/calibration_data/right-0037.png \
 --c ${root_dir}/calib_results1_stereo_refinement/stereo.yaml 
