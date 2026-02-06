from pathlib import Path

import cv2
from cam_calib_lib import ArgumentParsers as calib_args
from cam_calib_lib.ArgumentParsers import CameraCalibrationCharucoArgs
from cam_calib_lib.OpencvBoards import OpenCVCharucoBoard
from cam_calib_lib.StereoCalibration import stereo_calibration_pipeline
from cam_calib_lib.Utils import index_to_bgr
from cam_calib_lib import Utils as calib_utils
from cam_calib_lib.scripts.calibration.calibrate_stereo_camera import (
    display_undistorted_images,
    load_stereo_images,
    rectify_and_display,
)

import numpy as np
import numpy.typing as npt


class CliArgs(CameraCalibrationCharucoArgs):
    print_debug: bool = False
    save_debug_images: bool = False


def prepare_stereo_charuco_data(
    board: OpenCVCharucoBoard,
    left_images: list[Path],
    right_images: list[Path],
) -> tuple[
    list[npt.NDArray[np.float32]],  # obj_points
    list[npt.NDArray[np.float32]],  # left_img_points
    list[npt.NDArray[np.float32]],  # right_img_points
    list[npt.NDArray[np.uint8]],  # left_valid_images
    list[npt.NDArray[np.uint8]],  # right_valid_images
]:
    """
    Prepare object points and image points for stereo calibration.

    Returns:
        (obj_points, left_img_points, right_img_points, valid_pairs)
    """

    obj_points: list[npt.NDArray[np.float32]] = []
    left_img_points: list[npt.NDArray[np.float32]] = []
    right_img_points: list[npt.NDArray[np.float32]] = []
    left_valid_images: list[npt.NDArray[np.uint8]] = []
    right_valid_images: list[npt.NDArray[np.uint8]] = []

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
        left_ret, left_corners, left_corners_ids = board.detect_corners(left_img)
        right_ret, right_corners, right_corners_ids = board.detect_corners(right_img)

        valid_corners_in_image = 0
        if left_ret and right_ret:
            # Get indices in left_corners_ids that are also present in right_corners_ids
            left_common_idx = np.isin(
                left_corners_ids.flatten(), right_corners_ids.flatten()
            ).nonzero()[0]
            left_corners_filtered = left_corners[left_common_idx]
            left_corners_ids_filtered = left_corners_ids[left_common_idx]

            right_common_idx = np.isin(
                right_corners_ids.flatten(), left_corners_ids.flatten()
            ).nonzero()[0]
            right_corners_filtered = right_corners[right_common_idx]
            right_corners_ids_filtered = right_corners_ids[right_common_idx]

            if right_corners_filtered.shape[0] > 10:
                objp_left = board.get_object_points(
                    left_corners_filtered, left_corners_ids_filtered
                )
                objp_right = board.get_object_points(
                    right_corners_filtered, right_corners_ids_filtered
                )

                assert objp_left.shape == objp_right.shape, (
                    "Object point arrays have different shapes"
                )
                assert np.allclose(objp_left, objp_right), (
                    "Object points for left and right images differ"
                )

                obj_points.append(objp_left)
                left_img_points.append(left_corners_filtered)
                right_img_points.append(right_corners_filtered)
                left_valid_images.append(left_img)
                right_valid_images.append(right_img)

                valid_corners_in_image = right_corners_filtered.shape[0]

        minimum_corners_required = 20
        if valid_corners_in_image > minimum_corners_required:
            print(
                f"  Pair {idx}: ✓ Both images detected. Added {valid_corners_in_image} object points"
            )
        else:
            print(
                f"  Pair {idx}: ✗ {valid_corners_in_image} corners detected. "
                + f"Minimum {minimum_corners_required} corners required"
            )

    print(
        f"\nSuccessfully detected corners in {len(obj_points)}/{len(left_images)} pairs"
    )
    return (
        obj_points,
        left_img_points,
        right_img_points,
        left_valid_images,
        right_valid_images,
    )


def save_detected_corners(
    output_path: Path,
    obj_points: list[npt.NDArray[np.float32]],
    left_img_points: list[npt.NDArray[np.float32]],
    right_img_points: list[npt.NDArray[np.float32]],
    left_valid_images: list[npt.NDArray[np.uint8]],
    right_valid_images: list[npt.NDArray[np.uint8]],
    radius: int = 4,
    thickness: int = -1,
):
    output_path.mkdir(parents=True, exist_ok=True)

    num_views = len(left_img_points)

    for i in range(num_views):
        imgL = left_valid_images[i].copy()
        imgR = right_valid_images[i].copy()

        if imgL.ndim == 2:
            imgL = cv2.cvtColor(imgL, cv2.COLOR_GRAY2BGR)
        if imgR.ndim == 2:
            imgR = cv2.cvtColor(imgR, cv2.COLOR_GRAY2BGR)

        ptsL = left_img_points[i].reshape(-1, 2)
        ptsR = right_img_points[i].reshape(-1, 2)

        assert ptsL.shape == ptsR.shape, "Left/right corner count mismatch"

        n_pts = ptsL.shape[0]

        for j in range(n_pts):
            color = index_to_bgr(j, n_pts)

            pL = tuple(np.round(ptsL[j]).astype(int))
            pR = tuple(np.round(ptsR[j]).astype(int))

            cv2.circle(imgL, pL, radius, color, thickness)
            cv2.circle(imgR, pR, radius, color, thickness)

            # if draw_indices:
            #     cv2.putText(
            #         imgL, str(j), (pL[0] + 5, pL[1] - 5),
            #         cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            #     )
            #     cv2.putText(
            #         imgR, str(j), (pR[0] + 5, pR[1] - 5),
            #         cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            #     )

        # Concatenate for easier visual inspection
        debug = np.hstack([imgL, imgR])

        out_file = output_path / f"corners_{i:03d}.png"
        cv2.imwrite(str(out_file), debug)


def main(args: CliArgs):
    assert isinstance(args.images_path, Path)
    assert isinstance(args.output_path, Path)
    assert args.images_path.exists(), f"Images path does not exist: {args.images_path}"

    # args.save(args.output_path / "input_parameters.yaml")

    calib_utils.save_calibration_parameters_charuco(
        args.output_path / "input_parameters.yaml",
        args.aruco_dict,
        args.squares_in_x_dim,
        args.squares_in_y_dim,
        args.square_size,
        args.marker_size,
        args.distortion_flags,
        args.stereo_flags,
    )

    ## Opencv Flags
    distortion_flags = calib_args.compose_cv2_flags(args.distortion_flags)
    stereo_flags = calib_args.compose_cv2_flags(args.stereo_flags)
    aruco_dict_flag = calib_args.aruco_dict_from_string_string_list(args.aruco_dict)

    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_flag)
    board = OpenCVCharucoBoard(
        squares_in_x_dim=args.squares_in_x_dim,
        squares_in_y_dim=args.squares_in_y_dim,
        square_length=args.square_size,
        marker_length=args.marker_size,
        aruco_dict=aruco_dict,
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
    (
        obj_points,
        left_img_points,
        right_img_points,
        left_valid_images,
        right_valid_images,
    ) = prepare_stereo_charuco_data(board, left_images, right_images)

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
            distortion_flags,
            stereo_flags,
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

    if args.save_debug_images:
        print("Saving detected corners...")
        save_detected_corners(
            args.output_path / "debug_images" / "detected_corners",
            obj_points,
            left_img_points,
            right_img_points,
            left_valid_images,
            right_valid_images,
        )

        print("Saving rectified images...")
        save_vis_path = args.output_path / "debug_images" / "rect_vis"
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
