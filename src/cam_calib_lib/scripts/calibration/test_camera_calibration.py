from pathlib import Path
import cv2
from tap import Tap
import yaml
import numpy as np

from cam_calib_lib.OpencvBoards import OpenCVChessboard
from cam_calib_lib.Utils import load_camera_calib
from cam_calib_lib.scripts.calibration.calibrate_single_camera import (
    calculate_reprojection_error,
    display_projected_corners,
    load_camera_images,
    prepare_calibration_data,
)


class CliArgs(Tap):
    images_path: Path
    camera_calibration: Path
    camera_name: str = "left"  # "left" or "right"
    corners_per_board_row: int = 8
    corners_per_board_column: int = 5
    square_size: float = 0.01  # in meters


def load_camera_calibration(yaml_path):
    with open(yaml_path, "r") as f:
        calib = yaml.safe_load(f)

    # Camera intrinsic matrix
    K = np.array(calib["camera_matrix"]["data"], dtype=np.float64).reshape(
        calib["camera_matrix"]["rows"], calib["camera_matrix"]["cols"]
    )

    # Distortion coefficients
    D = np.array(calib["distortion_coefficients"]["data"], dtype=np.float64)

    return K, D


def calculate_rvecs_tvecs(obj_points, img_points, K, D):
    rvecs = []
    tvecs = []
    for objp, imgp in zip(obj_points, img_points):
        # OpenCV expects (N,1,3) and (N,1,2) shapes
        objp = np.asarray(objp, dtype=np.float32)
        imgp = np.asarray(imgp, dtype=np.float32)
        if imgp.ndim == 2:
            imgp = imgp.reshape(-1, 1, 2)

        objp = np.asarray(objp, dtype=np.float32).reshape(-1, 1, 3)
        imgp = np.asarray(imgp, dtype=np.float32).reshape(-1, 1, 2)

        success, rvec, tvec = cv2.solvePnP(objp, imgp, K, D, flags=cv2.SOLVEPNP_IPPE)
        if not success:
            raise RuntimeError("❌ cv2.solvePnP failed for one of the images.")
        rvecs.append(rvec)
        tvecs.append(tvec)

    return rvecs, tvecs


def main(args: CliArgs):
    # K, D = load_camera_calibration(args.images_path/ f"{args.camera}.yaml")
    params_dict = load_camera_calib(args.camera_calibration)
    mtx = params_dict["camera_matrix"]
    dist = params_dict["dist_coeffs"]

    print(f"Camera Matrix:\n{mtx}")
    print(f"Distortion Coefficients:\n{dist}")

    regular_chessboard = OpenCVChessboard(
        corners_per_row=args.corners_per_board_row,
        corners_per_column=args.corners_per_board_column,
        square_size=args.square_size,
    )
    images = load_camera_images(args.images_path, args.camera_name)

    obj_points, img_points = prepare_calibration_data(regular_chessboard, images)

    rvecs, tvecs = calculate_rvecs_tvecs(obj_points, img_points, mtx, dist)

    print(f"tvecs[0]: {tvecs[0]}")
    print(f"rvecs[0]: {rvecs[0]}")

    projected_points, _ = cv2.projectPoints(
        obj_points[0], rvecs[0], tvecs[0], mtx, dist
    )
    projected_points = projected_points.reshape(-1, 2)
    print(f"obj_points[0]: {obj_points[0][0:5]}")
    print(f"projected_points: {projected_points[0:5]}")

    error_with_distortion = calculate_reprojection_error(
        obj_points, img_points, rvecs, tvecs, mtx, dist
    )
    error_without_distortion = calculate_reprojection_error(
        obj_points,
        img_points,
        rvecs,
        tvecs,
        mtx,
        np.zeros_like(dist),
    )

    print(f"Reprojection error (with distortion): {error_with_distortion:.4f} pixels")
    print(
        f"Reprojection error (without distortion): {error_without_distortion:.4f} pixels"
    )

    print(
        f"\nDisplaying projected vs ground truth corners for first {args.camera_name} image..."
    )
    first_img = cv2.imread(str(images[0]))

    if first_img is not None and len(obj_points) > 0:
        first_img = np.asarray(first_img, dtype=np.uint8)
        # Use the first valid image's data
        display_projected_corners(
            first_img,
            obj_points[0],
            img_points[0],
            rvecs[0],
            tvecs[0],
            mtx,
            dist,
            args.camera_name,
            use_distortion=True,
        )


if __name__ == "__main__":
    args = CliArgs().parse_args()
    main(args)
