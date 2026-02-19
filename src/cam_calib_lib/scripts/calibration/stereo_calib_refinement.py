"""
Performs stereo calibration and removes images with high rectification errors.
Repeat until converging to a set of images with low rectification errors.
"""

from dataclasses import dataclass, field
import re
from pathlib import Path
import shutil
from typing import List, Tuple, Optional
import cv2
import numpy as np
import numpy.typing as npt
from tap import Tap
from cam_calib_lib.OpencvBoards import OpenCVChessboard
from cam_calib_lib.StereoCalibration import RectificationMapsOutput, stereo_calibration_pipeline
from cam_calib_lib import Utils as calib_utils
from cam_calib_lib import ArgumentParsers as calib_args
from cam_calib_lib.Logger import setup_logging_from_json
import logging
from pyprojroot.here import here  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class GlobalVariables:
    global_list_of_valid_images: list[int] = field(default_factory=list)

    global_valid_corners: dict[
        int, tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
    ] = field(default_factory=dict)

    global_chessboard_object_points: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )

    global_images_to_filter: list[int] = field(default_factory=list)

    def reset_images_to_filter(self):
        self.global_images_to_filter = []


global_variables = GlobalVariables()


class CliArgs(Tap):
    images_path: Path
    output_path: Path
    corners_per_board_row: int
    corners_per_board_column: int
    square_size: float  # in meters
    distortion_flags: list[str] = ["CALIB_FIX_K3", "CALIB_ZERO_TANGENT_DIST"]
    stereo_flags: list[str] = ["CALIB_FIX_INTRINSIC"]

    def configure(self):
        self.add_argument(  # type: ignore
            "--distortion_flags",
            help="Extra configuration options for distortion model."
            + " Default: [CALIB_FIX_K3, CALIB_ZERO_TANGENT_DIST]",
            nargs="+",
        )
        self.add_argument(  # type: ignore
            "--stereo_flags",
            help=(
                "Extra configuration options for stereo calibration."
                + " Default: CALIB_FIX_INTRINSIC. For MockOR Microscope use: CALIB_USE_INTRINSIC_GUESS"
            ),
            nargs="+",
        )


def extract_image_id(image_name: str) -> int:
    match = re.search(r"-(\d+)\.png$", image_name)
    if match:
        image_id = match.group(1)
    else:
        raise ValueError(f"Could not extract image ID from filename: {image_name}")
    return int(image_id)


def move_images_from_id_list(
    id_list: list[int],
    source_dir: Path,
    destination_dir: Path,
    image_extension: str = ".png",
) -> None:

    destination_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    missing: list[str] = []

    for img_id in id_list:
        for prefix in ("left", "right"):
            filename = f"{prefix}-{img_id:04d}{image_extension}"
            src_path = source_dir / filename
            dst_path = destination_dir / filename

            if src_path.exists():
                shutil.move(str(src_path), str(dst_path))
                moved += 1
            else:
                missing.append(str(src_path))

    logger.info(f"Moved {moved} files.")
    if missing:
        logger.warning(f"Missing files: {missing}")


def load_stereo_images(images_path: Path) -> Tuple[List[Path], List[Path]]:
    """
    Load left and right image pairs from the folder.
    Expected naming format: left-xxxx.png and right-xxxx.png
    """
    logger.info(f"Loading stereo images from {images_path}")
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

    logger.debug(f"Found {len(left_images)} stereo image pairs")

    return left_images, right_images


def prepare_calibration_data(
    regular_chessboard: OpenCVChessboard,
    left_images: List[Path],
    right_images: List[Path],
) -> Tuple[
    list[npt.NDArray[np.float32]],
    list[npt.NDArray[np.float32]],
    list[npt.NDArray[np.float32]],
]:
    """
    Prepare object points and image points for stereo calibration.

    Returns:
        (obj_points, left_img_points, right_img_points, valid_pairs)
    """

    objp = regular_chessboard.get_object_points(corners=None, ids=None)  # type: ignore
    global global_chessboard_object_points
    global_chessboard_object_points = objp.copy()

    obj_points: List[npt.NDArray[np.float32]] = []
    left_img_points: List[npt.NDArray[np.float32]] = []
    right_img_points: List[npt.NDArray[np.float32]] = []

    logger.info("Detecting chessboard corners...")
    for left_path, right_path in zip(left_images, right_images):
        left_img = cv2.imread(str(left_path))
        right_img = cv2.imread(str(right_path))
        image_id = extract_image_id(left_path.name)

        if left_img is None or right_img is None:  # type: ignore
            logger.warning(f"Warning: Could not load image pair {image_id}")
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
            assert left_corners is not None
            assert right_corners is not None

            obj_points.append(objp)
            left_img_points.append(left_corners)
            right_img_points.append(right_corners)
            logger.debug(f"  Pair {image_id}: ✓ Both images detected")

            global_variables.global_list_of_valid_images.append(image_id)
            global_variables.global_valid_corners[image_id] = (
                left_corners,
                right_corners,
            )

        else:
            status: list[str] = []
            if not left_ret:
                status.append("left failed")
            if not right_ret:
                status.append("right failed")
            logger.warning(f"  Pair {image_id}: ✗ {' and '.join(status)}")
            global_variables.global_images_to_filter.append(image_id)

    logger.info(
        f"Detected corners in {len(obj_points)}/{len(left_images)} pairs"
    )
    return obj_points, left_img_points, right_img_points


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


def report_rectification_errors(
    all_rect_ids: list[str],
    errors_per_image: list[float],
    output_path: Path,
):

    logger.info("Images with big errors:")
    images_to_filter: list[str] = []
    for idx, err in zip(all_rect_ids, errors_per_image):
        if err > 1.5:
            logger.info(f"Image {idx} error: {err}")
            images_to_filter.append(idx)

    rectified_error = np.array(errors_per_image).squeeze()

    # # Save images to filter in a txt file
    # with open(output_path / "images_to_filter.txt", "w") as f:
    #     for img_id in images_to_filter:
    #         f.write(f"{img_id}\n")

    ## Calculate error statistics
    logger.info(f"Rectified error mean: {np.mean(rectified_error)}")
    logger.info(f"Rectified error std: {np.std(rectified_error)}")
    logger.info(f"Rectified error max: {np.max(rectified_error)}")
    logger.info(f"Rectified error min: {np.min(rectified_error)}")
    logger.info(f"95% confidence interval: {np.percentile(rectified_error, 95)}")
 
def get_valid_images():
    obj_points: List[npt.NDArray[np.float32]] = []
    left_img_points: List[npt.NDArray[np.float32]] = []
    right_img_points: List[npt.NDArray[np.float32]] = []

    for image_id in global_variables.global_list_of_valid_images:
        left_corners, right_corners = global_variables.global_valid_corners[image_id]
        obj_points.append(global_chessboard_object_points)
        left_img_points.append(left_corners)
        right_img_points.append(right_corners)

    return obj_points, left_img_points, right_img_points


def calculate_final_rectification_error(rectification_maps_output: RectificationMapsOutput, regular_chessboard: OpenCVChessboard, images_path: Path):
    left_images, right_images = load_stereo_images(images_path)

    error_list: list[float] = []

    for left_path, right_path in zip(left_images, right_images):
        left_img = cv2.imread(str(left_path))
        right_img = cv2.imread(str(right_path))
        image_id = extract_image_id(left_path.name)

        left_rectified, right_rectified = rectify_and_display(
            left_img,  # type: ignore
            right_img,  # type: ignore
            rectification_maps_output.left_map1,
            rectification_maps_output.left_map2,
            rectification_maps_output.right_map1,
            rectification_maps_output.right_map2,
            display=False,  # No display
            save_vis_path=None,  # No saving
        )

        ## Calculate corners again to measure rectification error
        ret_left, corners_left, _ = regular_chessboard.detect_corners(
            left_rectified
        )
        ret_right, corners_right, _ = regular_chessboard.detect_corners(
            right_rectified
        )

        if ret_left and ret_right:
            assert corners_left is not None
            assert corners_right is not None

            d = corners_left[:, :, 1] - corners_right[:, :, 1]
            img_error = float(np.mean(np.sqrt(np.square(d))))  # type: ignore
            error_list.append(img_error)
        else:
            logger.warning(
                f"failed to detect corners in final rectified image {image_id}"
            )

    error_list_np = np.array(error_list)

    ## Calculate error statistics
    logger.info(f"Final rectification error statistics: (N={len(error_list_np)})")
    logger.debug(f"Number of images: {len(left_images)}")
    logger.info(f"Rectified error mean: {np.mean(error_list_np)}")
    logger.info(f"Rectified error std: {np.std(error_list_np)}")
    logger.info(f"Rectified error max: {np.max(error_list_np)}")
    logger.info(f"Rectified error min: {np.min(error_list_np)}")
    logger.info(f"95% confidence interval: {np.percentile(error_list_np, 95)}")


def main(args: CliArgs):
    assert isinstance(args.images_path, Path)
    assert isinstance(args.output_path, Path)
    assert args.images_path.exists(), f"Images path does not exist: {args.images_path}"
    args.output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Starting stereo calibration refinement...")
    logger.info(f"Images path: {args.images_path}")

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

    if first_img is None:  # type: ignore
        raise ValueError(f"Could not load first image: {left_images[0]}")

    image_size = (first_img.shape[1], first_img.shape[0])

    # Prepare calibration data
    obj_points, left_img_points, right_img_points = prepare_calibration_data(
        regular_chessboard, left_images, right_images
    )

    if len(obj_points) < 10:
        logger.warning(
            f"Warning: Only {len(obj_points)} valid pairs found. At least 10 pairs are recommended for good calibration."
        )

    move_images_from_id_list(
        global_variables.global_images_to_filter,
        args.images_path,
        args.images_path / "images_to_filter",
    )

    ## Calculate stereo calibration
    ## Repeat until converging to a set of images with low rectification errors.
    thresholds = [5.0, 4.0, 3.0, 2.0, 1.5]

    rectification_maps_output: Optional[RectificationMapsOutput] = None

    logger.info("Start refining stage... \n\n\n")
    for iteration in range(5):
        threshold = thresholds[iteration]
        logger.info(f"starting iteration {iteration} with threshold {threshold}\n\n")
        logger.info(
            f"Calibrate with {len(global_variables.global_list_of_valid_images)} valid images"
        )
        obj_points, left_img_points, right_img_points = get_valid_images()

        _, _, rectification_maps_output = stereo_calibration_pipeline(
            obj_points,
            left_img_points,
            right_img_points,
            image_size,
            distortion_flags,
            stereo_flags,
            args.output_path,
        )
        logger.info("\n")
        logger.info(f"Filter images with rectification error > {threshold} pixels\n")

        logger.info("Loading reload images...")
        left_images, right_images = load_stereo_images(args.images_path)

        global_variables.reset_images_to_filter()

        for left_path, right_path in zip(left_images, right_images):
            left_img = cv2.imread(str(left_path))
            right_img = cv2.imread(str(right_path))
            image_id = extract_image_id(left_path.name)

            left_rectified, right_rectified = rectify_and_display(
                left_img,  # type: ignore
                right_img,  # type: ignore
                rectification_maps_output.left_map1,
                rectification_maps_output.left_map2,
                rectification_maps_output.right_map1,
                rectification_maps_output.right_map2,
                display=False,  # No display
                save_vis_path=None,  # No saving
            )

            ## Calculate corners again to measure rectification error
            ret_left, corners_left, _ = regular_chessboard.detect_corners(
                left_rectified
            )
            ret_right, corners_right, _ = regular_chessboard.detect_corners(
                right_rectified
            )

            if ret_left and ret_right:
                assert corners_left is not None
                assert corners_right is not None

                d = corners_left[:, :, 1] - corners_right[:, :, 1]
                img_error = float(np.mean(np.sqrt(np.square(d))))  # type: ignore
                if img_error > threshold:
                    logger.info(f"Image {image_id} error: {img_error}")
                    global_variables.global_images_to_filter.append(image_id)
                    global_variables.global_list_of_valid_images.remove(image_id)
            else:
                logger.debug(
                    f"failed to detect corners in rectified image {image_id}"
                )
                global_variables.global_images_to_filter.append(image_id)
                global_variables.global_list_of_valid_images.remove(image_id)

        move_images_from_id_list(
            global_variables.global_images_to_filter,
            args.images_path,
            args.images_path / "images_to_filter",
        )
        logger.info(f"finished iteration {iteration}")
    
    assert rectification_maps_output is not None

    logger.info("Final evaluation")
    calculate_final_rectification_error(
        rectification_maps_output,
        regular_chessboard,
        args.images_path,
    )

    calib_utils.convert_yaml_to_json(args.output_path)

if __name__ == "__main__":
    args = CliArgs().parse_args()
    logger_config = here() / "configs" / "logger_config.json"
    logger_output_file = args.output_path / "stereo_calib_refinement.log"
    setup_logging_from_json(logger_config, logger_output_file)

    main(args)
