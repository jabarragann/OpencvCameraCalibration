#!/usr/bin/env bash

set -x

pydebug() {
    python -m debugpy --listen 5678 --wait-for-client "$@"
}

root_dir=/home/juan95/research/surpass_grant/soft_tissue_deformations/data/phantom_demo/calibration
python src/cam_calib_lib/scripts/stereo_tools/rectify_image_gui.py \
 --l ${root_dir}/calibration_images/left-0022.png \
 --r ${root_dir}/calibration_images/right-0022.png \
 --c ${root_dir}/offline_calibration/stereo.yaml 
