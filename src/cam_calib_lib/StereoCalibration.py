from dataclasses import dataclass
from pathlib import Path
import numpy as np
import numpy.typing as npt
import cv2

from cam_calib_lib.CalibrationErrorMetrics import (
    calculate_reprojection_error,
    epipolar_error_px,
)
from cam_calib_lib.Utils import (
    save_calibration_yaml_stereo_cam,
    save_calibration_yaml_stereo_cam_additional,
)

@dataclass
class StereoCalibrationOutput:
    left_mtx: npt.NDArray[np.float32]
    left_dist: npt.NDArray[np.float32]
    right_mtx: npt.NDArray[np.float32]
    right_dist: npt.NDArray[np.float32]
    R: npt.NDArray[np.float32]
    T: npt.NDArray[np.float32]
    E: npt.NDArray[np.float32]
    F: npt.NDArray[np.float32]


@dataclass
class RectificationOutput:
    R_left: npt.NDArray[np.float32]
    R_right: npt.NDArray[np.float32]
    P_left: npt.NDArray[np.float32]
    P_right: npt.NDArray[np.float32]
    left_roi: npt.NDArray[np.float32]
    right_roi: npt.NDArray[np.float32]


@dataclass
class RectificationMapsOutput:
    left_map1: npt.NDArray[np.float32]
    left_map2: npt.NDArray[np.float32]
    right_map1: npt.NDArray[np.float32]
    right_map2: npt.NDArray[np.float32]


def save_calibration_results(
    output_path: Path,
    left_camera_matrix: npt.NDArray[np.float32],
    left_dist_coeffs: npt.NDArray[np.float32],
    right_camera_matrix: npt.NDArray[np.float32],
    right_dist_coeffs: npt.NDArray[np.float32],
    R: npt.NDArray[np.float32],
    T: npt.NDArray[np.float32],
    E: npt.NDArray[np.float32],
    F: npt.NDArray[np.float32],
    image_size: tuple[int, int],
    reprojection_error: float,
) -> None:
    """Save calibration results to numpy files."""
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(str(output_path / "left_camera_matrix.npy"), left_camera_matrix)
    np.save(str(output_path / "left_dist_coeffs.npy"), left_dist_coeffs)
    np.save(str(output_path / "right_camera_matrix.npy"), right_camera_matrix)
    np.save(str(output_path / "right_dist_coeffs.npy"), right_dist_coeffs)
    np.save(str(output_path / "R.npy"), R)
    np.save(str(output_path / "T.npy"), T)
    np.save(str(output_path / "E.npy"), E)
    np.save(str(output_path / "F.npy"), F)

    # Save metadata
    metadata = {
        "image_size": image_size,
        "reprojection_error": reprojection_error,
    }
    np.save(str(output_path / "metadata.npy"), metadata)

    print(f"\nCalibration results saved to {output_path}")


def perform_stereo_calibration(
    obj_points: list[npt.NDArray[np.float32]],
    left_img_points: list[npt.NDArray[np.float32]],
    right_img_points: list[npt.NDArray[np.float32]],
    image_size: tuple[int, int],
    flags: int = cv2.CALIB_RATIONAL_MODEL,
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    float,
]:
    """
    Perform stereo calibration.

    Returns:
        (left_camera_matrix, left_dist_coeffs, right_camera_matrix, right_dist_coeffs,
         R, T, E, F, reprojection_error)
    """
    print("\nPerforming stereo calibration...")

    # First calibrate individual cameras
    flags = None
    print("Calibrating left camera...")
    ret_left, left_camera_matrix, left_dist_coeffs, left_rvecs, left_tvecs = (
        cv2.calibrateCamera(
            obj_points, left_img_points, image_size, None, None, flags=flags
        )
    )

    print("Calibrating right camera...")
    ret_right, right_camera_matrix, right_dist_coeffs, right_rvecs, right_tvecs = (
        cv2.calibrateCamera(
            obj_points, right_img_points, image_size, None, None, flags=flags
        )
    )

    # Stereo calibration
    stereo_calibration_flags = cv2.CALIB_FIX_INTRINSIC
    print("Performing stereo calibration...")
    (
        ret_stereo,
        left_camera_matrix,
        left_dist_coeffs,
        right_camera_matrix,
        right_dist_coeffs,
        R,
        T,
        E,
        F,
    ) = cv2.stereoCalibrate(
        obj_points,
        left_img_points,
        right_img_points,
        left_camera_matrix,
        left_dist_coeffs,
        right_camera_matrix,
        right_dist_coeffs,
        image_size,
        flags=stereo_calibration_flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
    )

    ## Custom error metrics
    errors_stereo, stats_stereo = epipolar_error_px(
        left_img_points, right_img_points, F
    )
    error_left_camera = calculate_reprojection_error(
        obj_points,
        left_img_points,
        left_rvecs,
        left_tvecs,
        left_camera_matrix,
        left_dist_coeffs,
    )
    error_right_camera = calculate_reprojection_error(
        obj_points,
        right_img_points,
        right_rvecs,
        right_tvecs,
        right_camera_matrix,
        right_dist_coeffs,
    )

    print("\nMy own error statistics")
    print(f"Epipolar error stereo: {stats_stereo['mean_px']:.4f} pixels")
    print(f"Epipolar error 95th percentile: {stats_stereo['p95_px']:.4f} pixels")
    print(f"Reprojection error left camera: {error_left_camera:.4f} pixels")
    print(f"Reprojection error right camera: {error_right_camera:.4f} pixels")

    print("\nOpencv Returned errors")
    print(f"Reprojection error left camera: {ret_left:.4f} pixels")
    print(f"Reprojection error right camera: {ret_right:.4f} pixels")
    print(f"Reprojection error stereo: {ret_stereo:.4f} pixels")

    # print(f"\nLeft camera matrix:\n{left_camera_matrix}")
    # print(f"\nRight camera matrix:\n{right_camera_matrix}")
    # print(f"\nRotation matrix:\n{R}")
    # print(f"\nTranslation vector:\n{T}")

    return (
        left_camera_matrix,
        left_dist_coeffs,
        right_camera_matrix,
        right_dist_coeffs,
        R,
        T,
        E,
        F,
        ret_stereo,
    )


def compute_rectification_matrices(
    left_camera_matrix: npt.NDArray[np.float32],
    left_dist_coeffs: npt.NDArray[np.float32],
    right_camera_matrix: npt.NDArray[np.float32],
    right_dist_coeffs: npt.NDArray[np.float32],
    image_size: tuple[int, int],
    R: npt.NDArray[np.float32],
    T: npt.NDArray[np.float32],
    alpha: float = 1.0,
):
    R_left, R_right, P_left, P_right, Q, left_roi, right_roi = cv2.stereoRectify(
        left_camera_matrix,
        left_dist_coeffs,
        right_camera_matrix,
        right_dist_coeffs,
        image_size,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=alpha,  # 0=valid pixels only, 1=all pixels
    )

    return R_left, R_right, P_left, P_right, Q, left_roi, right_roi


def compute_rectification_maps(
    R1: npt.NDArray[np.float32],
    R2: npt.NDArray[np.float32],
    P1: npt.NDArray[np.float32],
    P2: npt.NDArray[np.float32],
    image_size: tuple[int, int],
    left_mtx: npt.NDArray[np.float32],
    left_dist: npt.NDArray[np.float32],
    right_mtx: npt.NDArray[np.float32],
    right_dist: npt.NDArray[np.float32],
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    """
    Compute rectification maps for stereo images.

    Returns:
        (left_map1, left_map2, right_map1, right_map2, left_roi, right_roi)
    """
    print("\nComputing rectification maps...")

    # Compute rectification maps
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        left_mtx, left_dist, R1, P1, image_size, cv2.CV_16SC2
    )
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        right_mtx, right_dist, R2, P2, image_size, cv2.CV_16SC2
    )

    return left_map1, left_map2, right_map1, right_map2


def stereo_calibration_pipeline(
    obj_points, left_img_points, right_img_points, image_size, flags, output_path: Path
) -> tuple[StereoCalibrationOutput, RectificationOutput, RectificationMapsOutput]:
    # Perform stereo calibration
    (
        left_camera_matrix,
        left_dist_coeffs,
        right_camera_matrix,
        right_dist_coeffs,
        R,
        T,
        E,
        F,
        reprojection_error,
    ) = perform_stereo_calibration(
        obj_points, left_img_points, right_img_points, image_size, flags
    )

    # Compute rectification matrices
    R_left, R_right, P_left, P_right, _, left_roi, right_roi = (
        compute_rectification_matrices(
            left_camera_matrix,
            left_dist_coeffs,
            right_camera_matrix,
            right_dist_coeffs,
            image_size,
            R,
            T,
        )
    )

    # Compute rectification maps
    left_map1, left_map2, right_map1, right_map2 = compute_rectification_maps(
        R_left,
        R_right,
        P_left,
        P_right,
        image_size,
        left_camera_matrix,
        left_dist_coeffs,
        right_camera_matrix,
        right_dist_coeffs,
    )

    # Save calibration results
    save_calibration_results(
        output_path / "raw_results",
        left_camera_matrix,
        left_dist_coeffs,
        right_camera_matrix,
        right_dist_coeffs,
        R,
        T,
        E,
        F,
        image_size,
        reprojection_error,
    )

    ## Save into yaml files
    save_calibration_yaml_stereo_cam(
        filename=str(output_path / "left.yaml"),
        image_size=image_size,
        mtx=left_camera_matrix,
        dist=left_dist_coeffs,
        camera_name="left",
        rect_mtx=R_left,
        proj_mtx=P_left,
    )
    save_calibration_yaml_stereo_cam(
        filename=str(output_path / "right.yaml"),
        image_size=image_size,
        mtx=right_camera_matrix,
        dist=right_dist_coeffs,
        camera_name="right",
        rect_mtx=R_right,
        proj_mtx=P_right,
    )
    save_calibration_yaml_stereo_cam_additional(
        filename=str(output_path / "stereo.yaml"),
        image_size=image_size,
        F=F,
        E=E,
        T=T,
        R=R,
        left_mtx=left_camera_matrix,
        left_dist=left_dist_coeffs,
        right_mtx=right_camera_matrix,
        right_dist=right_dist_coeffs,
    )

    stereo_calibration_output = StereoCalibrationOutput(
        left_mtx=left_camera_matrix,
        left_dist=left_dist_coeffs,
        right_mtx=right_camera_matrix,
        right_dist=right_dist_coeffs,
        R=R,
        T=T,
        E=E,
        F=F,
    )
    rectification_output = RectificationOutput(
        R_left=R_left,
        R_right=R_right,
        P_left=P_left,
        P_right=P_right,
        left_roi=left_roi,
        right_roi=right_roi,
    )
    rectification_maps_output = RectificationMapsOutput(
        left_map1=left_map1,
        left_map2=left_map2,
        right_map1=right_map1,
        right_map2=right_map2,
    )

    return (
        stereo_calibration_output,
        rectification_output,
        rectification_maps_output,
    )
