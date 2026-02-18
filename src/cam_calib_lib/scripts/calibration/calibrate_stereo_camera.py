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
from cam_calib_lib import Utils as calib_utils
from cam_calib_lib import ArgumentParsers as calib_args


class CliArgs(Tap):
    images_path: Path
    output_path: Path = Path("./stereo_calibration_output")
    corners_per_board_row: int
    corners_per_board_column: int
    square_size: float  # in meters
    distortion_flags: list[str] = ["CALIB_FIX_K3", "CALIB_ZERO_TANGENT_DIST"]
    stereo_flags: list[str] = ["CALIB_FIX_INTRINSIC"]

    def configure(self):
        self.add_argument( # type: ignore 
            "--distortion_flags",
            help="Extra configuration options for distortion model."
            + " Default: [CALIB_FIX_K3, CALIB_ZERO_TANGENT_DIST]",
            nargs="+",
        )
        self.add_argument( # type: ignore
            "--stereo_flags",
            help=(
                "Extra configuration options for stereo calibration."
                + " Default: CALIB_FIX_INTRINSIC. For MockOR Microscope use: CALIB_USE_INTRINSIC_GUESS"
            ),
            nargs="+",
        )


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
    left_dict: dict[int, Path] = {}
    right_dict: dict[int, Path] = {}

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
    common_frames: list[int] = sorted(set(left_dict.keys()) & set(right_dict.keys()))

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
    list[npt.NDArray[np.float32]],
    list[npt.NDArray[np.float32]],
    list[npt.NDArray[np.float32]],
    list[tuple[int, int]],
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
            left_img_points.append(left_corners) # type: ignore
            right_img_points.append(right_corners) # type: ignore
            valid_pairs.append((idx, idx))
            print(f"  Pair {idx}: ✓ Both images detected")
        else:
            status: list[str] = []
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
    left_map1: npt.NDArray[np.float32],
    left_map2: npt.NDArray[np.float32],
    right_map1: npt.NDArray[np.float32],
    right_map2: npt.NDArray[np.float32],
    window_name: str = "Stereo Rectified Images",
    display: bool = True,
    save_vis_path: Optional[Path] = None,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
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
    left_rectified_display = left_rectified.copy()
    right_rectified_display = right_rectified.copy()

    # Draw horizontal lines for visual verification
    h, w = left_rectified.shape[:2]
    for idx, y in enumerate(range(0, h, 50)):
        if idx % 2 == 0:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        cv2.line(left_rectified_display, (0, y), (w, y), color, 1)
        cv2.line(right_rectified_display, (0, y), (w, y), color, 1)

    # Concatenate images side by side
    stereo_pair = np.hstack([left_rectified_display, right_rectified_display])

    if save_vis_path is not None:
        cv2.imwrite(str(save_vis_path), stereo_pair)

    # Display
    if display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, stereo_pair)
        print("Displaying rectified images. Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    left_rectified = np.asarray(left_rectified, dtype=np.uint8)
    right_rectified = np.asarray(right_rectified, dtype=np.uint8)

    return left_rectified, right_rectified


def display_undistorted_images(
    left_img: npt.NDArray[np.uint8],
    right_img: npt.NDArray[np.uint8],
    left_mtx: npt.NDArray[np.float32],
    left_dist: npt.NDArray[np.float32],
    right_mtx: npt.NDArray[np.float32],
    right_dist: npt.NDArray[np.float32],
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


def report_rectification_errors(
    all_rect_ids: list[str],
    errors_per_image: list[float],
    output_path: Path,
):

    print("Images with big errors:")
    images_to_filter:list[str] = []
    for idx, err in zip(all_rect_ids, errors_per_image):
        if err > 1.0:
            print(f"Image {idx} error: {err}")
            images_to_filter.append(idx)

    rectified_error = np.array(errors_per_image).squeeze()

    # Save images to filter in a txt file
    with open(output_path / "images_to_filter.txt", "w") as f:
        for img_id in images_to_filter:
            f.write(f"{img_id}\n")

    ## Calculate error statistics
    print(f"Rectified error mean: {np.mean(rectified_error)}")
    print(f"Rectified error std: {np.std(rectified_error)}")
    print(f"Rectified error max: {np.max(rectified_error)}")
    print(f"Rectified error min: {np.min(rectified_error)}")
    print(f"95% confidence interval: {np.percentile(rectified_error, 95)}")


def main(args: CliArgs):
    assert isinstance(args.images_path, Path)
    assert isinstance(args.output_path, Path)
    assert args.images_path.exists(), f"Images path does not exist: {args.images_path}"
    args.output_path.mkdir(parents=True, exist_ok=True)

    calib_utils.save_calibration_parameters_regular(
        args.output_path / "input_parameters.yaml",
        args.corners_per_board_row,
        args.corners_per_board_column,
        args.square_size,
        args.distortion_flags,
        args.stereo_flags,
    )

    ## Opencv Flags
    distortion_flags = calib_args.compose_cv2_flags(args.distortion_flags)
    stereo_flags = calib_args.compose_cv2_flags(args.stereo_flags)

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
    obj_points, left_img_points, right_img_points, _= (
        prepare_calibration_data(regular_chessboard, left_images, right_images)
    )

    if len(obj_points) < 10:
        print(
            f"Warning: Only {len(obj_points)} valid pairs found. At least 10 pairs are recommended for good calibration."
        )

    stereo_calibration_output, _, rectification_maps_output = (
        stereo_calibration_pipeline(
            obj_points,
            left_img_points,
            right_img_points,
            image_size,
            distortion_flags,
            stereo_flags,
            args.output_path,
        )
    )

    # Test rectification on first pair
    print("\nTesting rectification on first image pair...")
    left_img = cv2.imread(str(left_images[0]))
    right_img = cv2.imread(str(right_images[0]))

    if left_img is not None and right_img is not None: # type: ignore
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

    all_rect_corners_left: list[npt.NDArray[np.float32]] = []
    all_rect_corners_right: list[npt.NDArray[np.float32]] = []
    all_rect_ids: list[str] = []
    save_vis_path = args.output_path / "rect_vis"
    save_vis_path.mkdir(parents=True, exist_ok=True)
    for i in range(len(left_images)):
        left_img = cv2.imread(str(left_images[i]))
        right_img = cv2.imread(str(right_images[i]))

        match = re.search(r"-(\d+)\.png$", left_images[i].name)
        if match:
            image_id = match.group(1)
        else:
            raise ValueError(
                f"Could not extract image ID from filename: {left_images[i].name}"
            )

        left_rectified, right_rectified = rectify_and_display(
            left_img, # type: ignore
            right_img, # type: ignore
            rectification_maps_output.left_map1,
            rectification_maps_output.left_map2,
            rectification_maps_output.right_map1,
            rectification_maps_output.right_map2,
            display=False,
            save_vis_path=save_vis_path / f"rectified_stereo_pair_{i}.png",
        )

        ## Calculate corners again to measure rectification error
        ret_left, corners_left, _ = regular_chessboard.detect_corners(left_rectified)
        ret_right, corners_right, _ = regular_chessboard.detect_corners(right_rectified)

        if ret_left and ret_right:
            all_rect_corners_left.append(np.expand_dims(corners_left, axis=0)) # type: ignore
            all_rect_corners_right.append(np.expand_dims(corners_right, axis=0)) # type: ignore
            all_rect_ids.append(image_id)

    ## lrect_corners shape: (N_img, N_corners, 1, 2)
    lrect_corners = np.concatenate(all_rect_corners_left)
    rrect_corners = np.concatenate(all_rect_corners_right)

    ## Rectification error - Measure difference in y-direction
    ## For corresponding points.
    d = lrect_corners[:, :, :, 1] - rrect_corners[:, :, :, 1]
    errors_per_image: list[float] = []

    for idx in range(d.shape[0]):
        img_error = float(np.mean(np.sqrt(np.square(d[idx])))) # type: ignore
        errors_per_image.append(img_error)
        # print(f"Image {all_rect_ids[idx]} error: {img_error}")

    report_rectification_errors(all_rect_ids, errors_per_image, args.output_path)


if __name__ == "__main__":
    args = CliArgs().parse_args()
    main(args)
