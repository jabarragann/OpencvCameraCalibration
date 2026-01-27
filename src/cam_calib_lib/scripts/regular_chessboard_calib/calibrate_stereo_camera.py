"""
Stereo camera calibration using regular chessboard pattern.
Performs stereo calibration and provides rectification functionality.
"""

import re
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
import numpy.typing as npt
from tap import Tap

from cam_calib_lib.OpencvBoards import OpenCVChessboard


class CliArgs(Tap):
    images_path: Path
    output_path: Path = Path("./stereo_calibration_output")
    corners_per_board_row: int = 8
    corners_per_board_column: int = 5
    square_size: float = 0.01  # in meters
    flags: int = cv2.CALIB_FIX_K3


def load_stereo_images(images_path: Path) -> Tuple[List[Path], List[Path]]:
    """
    Load left and right image pairs from the folder.
    Expected naming format: left-xxxx.png and right-xxxx.png
    """
    left_images: List[Path] = []
    right_images: List[Path] = []

    # Find all left and right images
    all_files = sorted(images_path.glob("*.png"))

    # Extract frame numbers from filenames
    left_dict = {}
    right_dict = {}

    for file in all_files:
        filename = file.name
        if filename.startswith("left-"):
            match = re.search(r"left-(\d+)\.png", filename)
            if match:
                frame_num = int(match.group(1))
                left_dict[frame_num] = file
        elif filename.startswith("right-"):
            match = re.search(r"right-(\d+)\.png", filename)
            if match:
                frame_num = int(match.group(1))
                right_dict[frame_num] = file

    # Match pairs by frame number
    common_frames = sorted(set(left_dict.keys()) & set(right_dict.keys()))

    for frame_num in common_frames:
        left_images.append(left_dict[frame_num])
        right_images.append(right_dict[frame_num])

    print(f"Found {len(left_images)} stereo image pairs")
    return left_images, right_images


def detect_chessboard_corners(
    image: npt.NDArray[np.uint8],
    pattern_size: Tuple[int, int],
) -> Tuple[bool, Optional[npt.NDArray[np.float32]]]:
    """
    Detect chessboard corners in an image.

    Args:
        image: Input image
        pattern_size: (rows, cols) of inner corners

    Returns:
        (success, corners) tuple
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_FAST_CHECK
        + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )

    if ret:
        # Refine corner positions
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return True, corners

    return False, None


def prepare_calibration_data(
    regular_chessboard: OpenCVChessboard,
    left_images: List[Path],
    right_images: List[Path],
) -> Tuple[
    List[npt.NDArray[np.float32]],
    List[npt.NDArray[np.float32]],
    List[npt.NDArray[np.float32]],
    List[npt.NDArray[np.float32]],
]:
    """
    Prepare object points and image points for stereo calibration.

    Returns:
        (obj_points, left_img_points, right_img_points, valid_pairs)
    """

    objp = regular_chessboard.create_chessboard_points()

    obj_points: List[npt.NDArray[np.float32]] = []
    left_img_points: List[npt.NDArray[np.float32]] = []
    right_img_points: List[npt.NDArray[np.float32]] = []
    valid_pairs: List[Tuple[int, int]] = []

    print("Detecting chessboard corners...")
    for idx, (left_path, right_path) in enumerate(zip(left_images, right_images)):
        left_img = cv2.imread(str(left_path))
        right_img = cv2.imread(str(right_path))

        if left_img is None or right_img is None:
            print(f"Warning: Could not load image pair {idx}")
            continue

        left_img = np.asarray(left_img, dtype=np.uint8)
        right_img = np.asarray(right_img, dtype=np.uint8)

        # Detect corners in both images
        left_ret, left_corners = detect_chessboard_corners(left_img, regular_chessboard.partern_size)
        right_ret, right_corners = detect_chessboard_corners(right_img, regular_chessboard.partern_size)

        # if left_ret:
        #     img = left_img.copy()
        #     cv2.drawChessboardCorners(img, pattern_size, left_corners, left_ret)
        #     cv2.namedWindow('debug_window', cv2.WINDOW_NORMAL)
        #     cv2.imshow('debug_window', img)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        if left_ret and right_ret:
            obj_points.append(objp)
            left_img_points.append(left_corners)
            right_img_points.append(right_corners)
            valid_pairs.append((idx, idx))
            print(f"  Pair {idx}: ✓ Both images detected")
        else:
            status = []
            if not left_ret:
                status.append("left failed")
            if not right_ret:
                status.append("right failed")
            print(f"  Pair {idx}: ✗ {' and '.join(status)}")

    print(
        f"\nSuccessfully detected corners in {len(obj_points)}/{len(left_images)} pairs"
    )
    return obj_points, left_img_points, right_img_points, valid_pairs


def perform_stereo_calibration(
    obj_points: List[npt.NDArray[np.float32]],
    left_img_points: List[npt.NDArray[np.float32]],
    right_img_points: List[npt.NDArray[np.float32]],
    image_size: Tuple[int, int],
    flags: int = cv2.CALIB_RATIONAL_MODEL,
) -> Tuple[
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
    _, left_camera_matrix, left_dist_coeffs, _, _ = cv2.calibrateCamera(
        obj_points, left_img_points, image_size, None, None, flags=flags
    )

    print("Calibrating right camera...")
    _, right_camera_matrix, right_dist_coeffs, _, _ = cv2.calibrateCamera(
        obj_points, right_img_points, image_size, None, None, flags=flags
    )

    # Stereo calibration
    print("Performing stereo calibration...")
    (
        ret,
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
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
    )

    print("\nStereo calibration completed!")
    print(f"Reprojection error: {ret:.4f} pixels")
    print(f"\nLeft camera matrix:\n{left_camera_matrix}")
    print(f"\nRight camera matrix:\n{right_camera_matrix}")
    print(f"\nRotation matrix:\n{R}")
    print(f"\nTranslation vector:\n{T}")

    return (
        left_camera_matrix,
        left_dist_coeffs,
        right_camera_matrix,
        right_dist_coeffs,
        R,
        T,
        E,
        F,
        ret,
    )


def compute_rectification_maps(
    left_camera_matrix: npt.NDArray[np.float32],
    left_dist_coeffs: npt.NDArray[np.float32],
    right_camera_matrix: npt.NDArray[np.float32],
    right_dist_coeffs: npt.NDArray[np.float32],
    image_size: Tuple[int, int],
    R: npt.NDArray[np.float32],
    T: npt.NDArray[np.float32],
) -> Tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    """
    Compute rectification maps for stereo images.

    Returns:
        (left_map1, left_map2, right_map1, right_map2, left_roi, right_roi)
    """
    print("\nComputing rectification maps...")

    R1, R2, P1, P2, Q, left_roi, right_roi = cv2.stereoRectify(
        left_camera_matrix,
        left_dist_coeffs,
        right_camera_matrix,
        right_dist_coeffs,
        image_size,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.9,  # 0=valid pixels only, 1=all pixels, 0.9=good balance
    )

    # Compute rectification maps
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        left_camera_matrix, left_dist_coeffs, R1, P1, image_size, cv2.CV_16SC2
    )
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        right_camera_matrix, right_dist_coeffs, R2, P2, image_size, cv2.CV_16SC2
    )

    print("Rectification maps computed!")
    return left_map1, left_map2, right_map1, right_map2, left_roi, right_roi


def rectify_and_display(
    left_img: npt.NDArray[np.uint8],
    right_img: npt.NDArray[np.uint8],
    left_map1: npt.NDArray,
    left_map2: npt.NDArray,
    right_map1: npt.NDArray,
    right_map2: npt.NDArray,
    window_name: str = "Stereo Rectified Images",
) -> None:
    """
    Rectify a pair of stereo images and display them side by side with OpenCV.

    Args:
        left_img: Left image
        right_img: Right image
        left_map1, left_map2: Left rectification maps
        right_map1, right_map2: Right rectification maps
        window_name: Name of the display window
    """
    # Rectify images
    left_rectified = cv2.remap(left_img, left_map1, left_map2, cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_img, right_map1, right_map2, cv2.INTER_LINEAR)

    # Draw horizontal lines for visual verification
    h, w = left_rectified.shape[:2]
    for y in range(0, h, 50):
        cv2.line(left_rectified, (0, y), (w, y), (0, 255, 0), 1)
        cv2.line(right_rectified, (0, y), (w, y), (0, 255, 0), 1)

    # Concatenate images side by side
    stereo_pair = np.hstack([left_rectified, right_rectified])
    # Make the window resizable before showing the image
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Optionally, you can resize the image to fit your screen, e.g. to max width 1600
    max_width = 1000
    height, width = stereo_pair.shape[:2]
    if width > max_width:
        scale = max_width / width
        stereo_pair = cv2.resize(
            stereo_pair, (max_width, int(height * scale)), interpolation=cv2.INTER_AREA
        )

    # Display
    cv2.imshow(window_name, stereo_pair)
    print("Displaying rectified images. Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def display_undistorted_images(
    left_img,
    right_img,
    left_camera_matrix,
    left_dist_coeffs,
    right_camera_matrix,
    right_dist_coeffs,
):
    """
    Display the undistorted left and right images using their respective intrinsics.
    """
    left_undistorted = cv2.undistort(left_img, left_camera_matrix, left_dist_coeffs)
    right_undistorted = cv2.undistort(right_img, right_camera_matrix, right_dist_coeffs)
    # left_undistorted = left_img.copy()
    # right_undistorted = right_img.copy()

    concatenated = np.hstack([left_undistorted, right_undistorted])
    # Concatenate images side by side for display
    cv2.namedWindow("Undistorted Left/Right Images", cv2.WINDOW_NORMAL)

    # max_width = 1000
    # h, w = concatenated.shape[:2]
    # if w > max_width:
    #     scale = max_width / w
    #     concatenated = cv2.resize(
    #         concatenated, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA
    #     )

    cv2.imshow("Undistorted Left/Right Images", concatenated)
    print("Showing undistorted left/right images. Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
    image_size: Tuple[int, int],
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


def main(args: CliArgs):
    assert isinstance(args.images_path, Path)
    assert isinstance(args.output_path, Path)
    assert args.images_path.exists(), f"Images path does not exist: {args.images_path}"

    regular_chessboard = OpenCVChessboard(
        corners_per_row=args.corners_per_board_row,
        corners_per_column=args.corners_per_board_column,
        square_size=args.square_size,
    )

    # Load stereo image pairs
    left_images, right_images = load_stereo_images(args.images_path)

    if len(left_images) == 0:
        raise ValueError("No valid stereo image pairs found!")

    # Get image size from first image
    first_img = cv2.imread(str(left_images[0]))
    if first_img is None:
        raise ValueError(f"Could not load first image: {left_images[0]}")
    image_size = (first_img.shape[1], first_img.shape[0])

    # Prepare calibration data
    obj_points, left_img_points, right_img_points, valid_pairs = (
        prepare_calibration_data(
            regular_chessboard, left_images, right_images
        )
    )

    if len(obj_points) < 10:
        print(
            f"Warning: Only {len(obj_points)} valid pairs found. At least 10 pairs are recommended for good calibration."
        )

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
        obj_points, left_img_points, right_img_points, image_size, args.flags
    )

    # Compute rectification maps
    left_map1, left_map2, right_map1, right_map2, left_roi, right_roi = (
        compute_rectification_maps(
            left_camera_matrix,
            left_dist_coeffs,
            right_camera_matrix,
            right_dist_coeffs,
            image_size,
            R,
            T,
        )
    )

    # Save rectification maps
    args.output_path.mkdir(parents=True, exist_ok=True)
    np.save(str(args.output_path / "left_map1.npy"), left_map1)
    np.save(str(args.output_path / "left_map2.npy"), left_map2)
    np.save(str(args.output_path / "right_map1.npy"), right_map1)
    np.save(str(args.output_path / "right_map2.npy"), right_map2)
    np.save(str(args.output_path / "left_roi.npy"), left_roi)
    np.save(str(args.output_path / "right_roi.npy"), right_roi)

    # Save calibration results
    save_calibration_results(
        args.output_path,
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

    # Test rectification on first pair
    print("\nTesting rectification on first image pair...")
    left_img = cv2.imread(str(left_images[0]))
    right_img = cv2.imread(str(right_images[0]))

    if left_img is not None and right_img is not None:
        left_img = np.asarray(left_img, dtype=np.uint8)
        right_img = np.asarray(right_img, dtype=np.uint8)

        display_undistorted_images(
            left_img,
            right_img,
            left_camera_matrix,
            left_dist_coeffs,
            right_camera_matrix,
            right_dist_coeffs,
        )

        rectify_and_display(
            left_img, right_img, left_map1, left_map2, right_map1, right_map2
        )

    print("\nCalibration complete!")


if __name__ == "__main__":
    args = CliArgs().parse_args()
    main(args)
