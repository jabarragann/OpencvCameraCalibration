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
from cam_calib_lib.StereoCalibration import stereo_calibration_pipeline


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
        if filename.startswith("left"):
            match = re.search(r"left[_-](\d+)\.png", filename)
            if match:
                frame_num = int(match.group(1))
                left_dict[frame_num] = file
        elif filename.startswith("right"):
            match = re.search(r"right[_-](\d+)\.png", filename)
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

    objp = regular_chessboard.get_object_points(corners=None, ids=None)  # type: ignore

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
        left_ret, left_corners, _ = regular_chessboard.detect_corners(left_img)
        right_ret, right_corners, _ = regular_chessboard.detect_corners(right_img)

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


def rectify_and_display(
    left_img: npt.NDArray[np.uint8],
    right_img: npt.NDArray[np.uint8],
    left_map1: npt.NDArray,
    left_map2: npt.NDArray,
    right_map1: npt.NDArray,
    right_map2: npt.NDArray,
    window_name: str = "Stereo Rectified Images",
    display: bool = True,
    save_vis_path: Optional[Path] = None,
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

    if save_vis_path is not None:
        cv2.imwrite(str(save_vis_path), stereo_pair)

    # Display
    if display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, stereo_pair)
        print("Displaying rectified images. Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def display_undistorted_images(
    left_img,
    right_img,
    left_mtx,
    left_dist,
    right_mtx,
    right_dist,
):
    """
    Display the undistorted left and right images using their respective intrinsics.
    """

    w, h = left_img.shape[:2]
    left_new_mtx, left_roi = cv2.getOptimalNewCameraMatrix(
        left_mtx, left_dist, (w, h), 1.0, (w, h)
    )
    right_new_mtx, right_roi = cv2.getOptimalNewCameraMatrix(
        right_mtx, right_dist, (w, h), 1.0, (w, h)
    )

    left_undistorted = cv2.undistort(left_img, left_mtx, left_dist, None, left_new_mtx)
    right_undistorted = cv2.undistort(
        right_img, right_mtx, right_dist, None, right_new_mtx
    )
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
        prepare_calibration_data(regular_chessboard, left_images, right_images)
    )

    if len(obj_points) < 10:
        print(
            f"Warning: Only {len(obj_points)} valid pairs found. At least 10 pairs are recommended for good calibration."
        )

    stereo_calibration_output, rectification_output, rectification_maps_output = (
        stereo_calibration_pipeline(
            obj_points,
            left_img_points,
            right_img_points,
            image_size,
            args.flags,
            args.output_path,
        )
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
            stereo_calibration_output.left_mtx,
            stereo_calibration_output.left_dist,
            stereo_calibration_output.right_mtx,
            stereo_calibration_output.right_dist,
        )

        rectify_and_display(
            left_img,
            right_img,
            rectification_maps_output.left_map1,
            rectification_maps_output.left_map2,
            rectification_maps_output.right_map1,
            rectification_maps_output.right_map2,
        )

    print("\nCalibration complete!")

    print("Saving rectified images...")
    save_vis_path = args.output_path / "rect_vis"
    save_vis_path.mkdir(parents=True, exist_ok=True)
    for i in range(len(left_images)):
        left_img = cv2.imread(str(left_images[i]))
        right_img = cv2.imread(str(right_images[i]))
        rectify_and_display(
            left_img,
            right_img,
            rectification_maps_output.left_map1,
            rectification_maps_output.left_map2,
            rectification_maps_output.right_map1,
            rectification_maps_output.right_map2,
            display=False,
            save_vis_path=save_vis_path / f"rectified_stereo_pair_{i}.png",
        )


if __name__ == "__main__":
    args = CliArgs().parse_args()
    main(args)
