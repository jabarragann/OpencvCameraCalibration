import cv2
import numpy as np
import numpy.typing as npt


def calculate_reprojection_error(
    obj_points: list[npt.NDArray[np.float32]],
    img_points: list[npt.NDArray[np.float32]],
    rvecs: list[npt.NDArray[np.float32]],
    tvecs: list[npt.NDArray[np.float32]],
    mtx: npt.NDArray[np.float32],
    dist: npt.NDArray[np.float32],
) -> float:
    """
    Calculate reprojection error for camera calibration.

    Args:
        obj_points: List of 3D object points
        img_points: List of 2D image points
        rvecs: List of rotation vectors
        tvecs: List of translation vectors
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients

    Returns:
        Mean reprojection error in pixels
    """
    mean_error = 0.0
    num_points = 0

    for i in range(len(obj_points)):
        imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
        num_points += 1

    mean_error /= len(obj_points)

    return mean_error


def epipolar_error_px(
    pts_left: list[npt.NDArray[np.float32]],
    pts_right: list[npt.NDArray[np.float32]],
    F: np.ndarray,
):
    """
    Compute epipolar point-to-line distances (in pixels).

    Parameters
    ----------
    pts_left : (N, 2) array
        Matched points in the left image (pixel coordinates).
    pts_right : (N, 2) array
        Corresponding matched points in the right image.
    F : (3, 3) array
        Fundamental matrix (left -> right).

    Returns
    -------
    errors : (N,) array
        Point-to-epipolar-line distances in pixels.
    stats : dict
        Mean, median, 95th percentile, max.
    """

    pts_left_concat = np.concatenate(pts_left, axis=0)
    pts_right_concat = np.concatenate(pts_right, axis=0)
    assert pts_left_concat.shape == pts_right_concat.shape
    assert pts_left_concat.shape[2] == 2

    # Convert to homogeneous coordinates
    pts_l = cv2.convertPointsToHomogeneous(pts_left_concat).squeeze()
    pts_r = cv2.convertPointsToHomogeneous(pts_right_concat).squeeze()

    # Epipolar lines in right image: l_r = F * x_l
    lines_r = (F @ pts_l.T).T  # (N, 3)

    # Distance from right points to their epipolar lines
    num = np.abs(np.sum(lines_r * pts_r, axis=1))
    den = np.sqrt(lines_r[:, 0] ** 2 + lines_r[:, 1] ** 2)

    errors = num / den

    stats = {
        "mean_px": float(np.mean(errors)),
        "median_px": float(np.median(errors)),
        "p95_px": float(np.percentile(errors, 95)),
        "max_px": float(np.max(errors)),
    }

    return errors, stats
