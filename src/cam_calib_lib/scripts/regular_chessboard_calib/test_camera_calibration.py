from pathlib import Path
import cv2
from tap import Tap
import yaml
import numpy as np

from cam_calib_lib.scripts.regular_chessboard_calib.calibrate_single_camera import calculate_reprojection_error, display_projected_corners, load_camera_images, prepare_calibration_data

class CliArgs(Tap):
    images_path: Path
    camera: str = "left"  # "left" or "right"
    chessboard_rows: int = 5
    chessboard_cols: int = 8
    square_size: float = 0.025  # in meters

def load_camera_calibration(yaml_path):
    with open(yaml_path, "r") as f:
        calib = yaml.safe_load(f)

    # Camera intrinsic matrix
    K = np.array(calib["camera_matrix"]["data"], dtype=np.float64).reshape(
        calib["camera_matrix"]["rows"],
        calib["camera_matrix"]["cols"]
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
            imgp = imgp.reshape(-1,1,2)
        success, rvec, tvec = cv2.solvePnP(objp, imgp, K, D)
        if not success:
            raise RuntimeError("cv2.solvePnP failed for one of the images.")
        rvecs.append(rvec)
        tvecs.append(tvec)

    return rvecs, tvecs

def main(args: CliArgs):
    K, D = load_camera_calibration(args.images_path/ f"{args.camera}.yaml")
    print(f"Camera Matrix:\n{K}")
    print(f"Distortion Coefficients:\n{D}")

    images = load_camera_images(args.images_path, args.camera)

    pattern_size = (args.chessboard_rows, args.chessboard_cols)
    obj_points, img_points = prepare_calibration_data(
        images, pattern_size, args.square_size
    )

    rvecs, tvecs = calculate_rvecs_tvecs(obj_points, img_points, K, D)

    error_with_distortion = calculate_reprojection_error(
        obj_points,
        img_points,
        rvecs,
        tvecs,
        K,
        D,
        use_distortion=True,
    )
    error_without_distortion = calculate_reprojection_error(
        obj_points,
        img_points,
        rvecs,
        tvecs,
        K,
        D,
        use_distortion=False,
    )

    print(f"Reprojection error (with distortion): {error_with_distortion:.4f} pixels")
    print(
        f"Reprojection error (without distortion): {error_without_distortion:.4f} pixels"
    )

    print(f"\nDisplaying projected vs ground truth corners for first {args.camera} image...")
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
            K,
            D,
            args.camera,
            use_distortion=True,
        )


if __name__ == "__main__":
    args = CliArgs().parse_args()
    main(args)