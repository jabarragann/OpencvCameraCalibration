from pathlib import Path

import cv2
import numpy as np
from tap import Tap

from cam_calib_lib.CalibrationErrorMetrics import calculate_reprojection_error
from cam_calib_lib.OpencvBoards import OpenCVChessboard, OpenCVCornerDetector
from cam_calib_lib.scripts.calibration.calibrate_stereo_camera import (
    load_stereo_images,
    rectify_and_display,
)
from cam_calib_lib.scripts.calibration.test_camera_calibration import (
    calculate_rvecs_tvecs,
)
from cam_calib_lib.StereoCalibration import compute_rectification_maps
from cam_calib_lib.Utils import load_camera_calib


class CliArgs(Tap):
    images_path: Path
    output_path: Path
    left_calibration_file: Path
    right_calibration_file: Path
    corners_per_board_row: int = 8
    corners_per_board_column: int = 6
    square_size: float = 0.004


def main(args: CliArgs):
    debug = False
    params_dict_left = load_camera_calib(args.left_calibration_file)
    parms_dict_right = load_camera_calib(args.right_calibration_file)

    mtx_left = params_dict_left["camera_matrix"]
    dist_left = params_dict_left["dist_coeffs"]
    rec_matrix_left = params_dict_left["rectification_matrix"]
    projection_matrix_left = params_dict_left["projection_matrix"]

    mtx_right = parms_dict_right["camera_matrix"]
    dist_right = parms_dict_right["dist_coeffs"]
    rec_matrix_right = parms_dict_right["rectification_matrix"]
    projection_matrix_right = parms_dict_right["projection_matrix"]
    image_size = params_dict_left["image_size"]

    left_map1, left_map2, right_map1, right_map2 = compute_rectification_maps(
        rec_matrix_left,
        rec_matrix_right,
        projection_matrix_left,
        projection_matrix_right,
        image_size,
        mtx_left,
        dist_left,
        mtx_right,
        dist_right,
    )

    left_images, right_images = load_stereo_images(args.images_path)
    save_vis_path = args.output_path / "rectified_stereo_pairs"
    save_vis_path.mkdir(parents=True, exist_ok=True)

    # chessboard: OpenCVCornerDetector = OpenCVChessboard(
    #     corners_per_row=args.corners_per_board_row,
    #     corners_per_column=args.corners_per_board_column,
    #     square_size=args.square_size,
    # )
    # all_obj_points = []
    # all_img_left_points = []
    # all_img_right_points = []

    for i in range(len(left_images)):
        print(f"Processing image {i} of {len(left_images)}")
        left_img = cv2.imread(str(left_images[i]))
        right_img = cv2.imread(str(right_images[i]))

        left_rectified, right_rectified = rectify_and_display(
            left_img,
            right_img,
            left_map1,
            left_map2,
            right_map1,
            right_map2,
            display=False,
            save_vis_path=save_vis_path / f"rectified_stereo_pair_{i}.png",
        )

        ## If corner detection needed -- Uncomment the following lines.
    #     ret_left, corners_left, ids_left = chessboard.detect_corners(left_rectified)
    #     ret_right, corners_right, ids_right = chessboard.detect_corners(right_rectified)

    #     obj_points_left = chessboard.get_object_points(corners_left, ids_left)
    #     obj_points_right = chessboard.get_object_points(corners_right, ids_right)

    #     assert np.allclose(obj_points_left, obj_points_right)

    #     if ret_left and ret_right:
    #         print(f"Found corners in left and right images {i}")
    #         all_obj_points.append(obj_points_left)
    #         all_img_left_points.append(corners_left)
    #         all_img_right_points.append(corners_right)

    #     if debug and i > 10:
    #         break

    # # obj_points, img_points = prepare_calibration_data(regular_chessboard, images)

    # ## Experiment 1: Reprojection error experiments
    # rvecs_left, tvecs_left = calculate_rvecs_tvecs(
    #     all_obj_points, all_img_left_points, mtx_left, dist_left
    # )

    # left_error = calculate_reprojection_error(
    #     all_obj_points, all_img_left_points, rvecs_left, tvecs_left, mtx_left, dist_left
    # )
    # print(f"Reprojection error (the wrong way) {left_error:.4f} pixels")

    # rvecs_left, tvecs_left = calculate_rvecs_tvecs(
    #     all_obj_points, all_img_left_points, projection_matrix_left[:3, :3], np.zeros_like(dist_left)
    # )
    # left_error2 = calculate_reprojection_error(
    #     all_obj_points,
    #     all_img_left_points,
    #     rvecs_left,
    #     tvecs_left,
    #     projection_matrix_left[:3, :3],
    #     np.zeros_like(dist_left),
    # )
    # print(f"Reprojection error (the right way) {left_error2:.4f} pixels")


if __name__ == "__main__":
    args = CliArgs().parse_args()
    main(args)
