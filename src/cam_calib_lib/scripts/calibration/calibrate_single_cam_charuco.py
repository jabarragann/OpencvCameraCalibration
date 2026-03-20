from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from cam_calib_lib.ArgumentParsers import CameraCalibrationCharucoArgs

# from cam_calib_lib.CornerDetector import CharucoBoard
from cam_calib_lib.OpencvBoards import OpenCVCharucoBoard
from cam_calib_lib.scripts.calibrate_camera import calculate_reprojection_error
from cam_calib_lib.scripts.calibration.calibrate_single_camera import (
    display_projected_corners,
    display_undistorted_image,
    perform_single_calibration,
    save_calibration_results,
)


class CliArgs(CameraCalibrationCharucoArgs):
    print_debug: bool = False


def load_camera_images(images_path: Path, camera_name: str) -> list[Path]:
    all_files: list[Path] = list(images_path.glob(f"{camera_name}_*.png"))
    all_files = sorted(all_files)
    return all_files


def prepare_calibration_data(
    chessboard: OpenCVCharucoBoard,
    images: list[Path],
    print_debug: bool = False,
) -> tuple[
    list[npt.NDArray[np.float32]],
    list[npt.NDArray[np.float32]],
]:
    """
    Prepare object points and image points for single camera calibration.

    Returns:
        (obj_points, img_points)
    """

    obj_points: list[npt.NDArray[np.float32]] = []
    img_points: list[npt.NDArray[np.float32]] = []

    print("Detecting chessboard corners...")
    for idx, img_path in enumerate(images):
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"Warning: Could not load image {idx}: {img_path}")
            continue

        img = np.asarray(img, dtype=np.uint8)

        # Detect corners
        ret, charuco_corners, charuco_ids = chessboard.detect_corners(img)

        failure = charuco_corners is None or charuco_ids is None or not ret

        if not failure:
            obj_points_i = chessboard.get_object_points(charuco_corners, charuco_ids)
            img_points_i = charuco_corners

            obj_points.append(obj_points_i)
            img_points.append(img_points_i)

        if not failure:
            img_copy = img.copy()
            cv2.aruco.drawDetectedCornersCharuco(img_copy, charuco_corners, charuco_ids)  # type: ignore[no-untyped-call]
            # cv2.namedWindow(f"Corners for image {img_path.name}", cv2.WINDOW_NORMAL)
            cv2.imshow(f"Corners for image {img_path.name}", img_copy)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if print_debug:
            if not failure:
                print(f"  Image {idx}: ✓ Corners detected")
            else:
                print(f"  Image {idx}: ✗ Failed to detect corners")

    print(f"\nSuccessfully detected corners in {len(obj_points)}/{len(images)} images")
    return obj_points, img_points


def main(args: CliArgs):
    assert isinstance(args.images_path, Path)
    assert isinstance(args.output_path, Path)
    assert args.images_path.exists(), f"Images path does not exist: {args.images_path}"
    assert args.camera_name in [
        "left",
        "right",
    ], f"Camera must be 'left' or 'right', got '{args.camera_name}'"

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

    charuco_board_7x10 = OpenCVCharucoBoard(
        squares_in_x_dim=args.squares_in_x_dim,
        squares_in_y_dim=args.squares_in_y_dim,
        aruco_dict=aruco_dict,
        square_length=args.square_size,
        marker_length=args.marker_size,
    )

    # Load images for the specified camera
    images = load_camera_images(args.images_path, args.camera_name)

    if len(images) == 0:
        raise ValueError(f"No valid images found for {args.camera_name} camera!")

    # Get image size from first image
    first_img = cv2.imread(str(images[0]))
    if first_img is None:
        raise ValueError(f"Could not load first image: {images[0]}")
    image_size = (first_img.shape[1], first_img.shape[0])

    # Prepare calibration data
    obj_points, img_points = prepare_calibration_data(
        charuco_board_7x10, images, print_debug=args.print_debug
    )

    if len(obj_points) < 10:
        print(
            f"Warning: Only {len(obj_points)} valid images found. At least 10 images are recommended for good calibration."
        )

    ## All the same here below.
    # Perform single camera calibration
    camera_matrix, dist_coeffs, rvecs, tvecs, reprojection_error = (
        perform_single_calibration(
            obj_points, img_points, image_size, args.distortion_flags
        )
    )

    # Calculate reprojection errors with and without distortion
    print("\nCalculating reprojection errors...")
    error_with_distortion = calculate_reprojection_error(
        obj_points,
        img_points,
        rvecs,
        tvecs,
        camera_matrix,
        dist_coeffs,
    )
    error_without_distortion = calculate_reprojection_error(
        obj_points,
        img_points,
        rvecs,
        tvecs,
        camera_matrix,
        np.zeros_like(dist_coeffs),
    )

    print(f"Reprojection error (with distortion): {error_with_distortion:.4f} pixels")
    print(
        f"Reprojection error (without distortion): {error_without_distortion:.4f} pixels"
    )

    # Save calibration results (using the error with distortion as the main error)
    save_calibration_results(
        args.output_path,
        camera_matrix,
        dist_coeffs,
        image_size,
        error_with_distortion,
        args.camera_name,
    )

    # Display projected vs ground truth corners for first image
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
            camera_matrix,
            dist_coeffs,
            args.camera_name,
            use_distortion=True,
        )

    # Test undistortion on first image
    print(f"\nTesting undistortion on first {args.camera_name} image...")
    img = cv2.imread(str(images[0]))

    if img is not None:
        img = np.asarray(img, dtype=np.uint8)
        display_undistorted_image(img, camera_matrix, dist_coeffs, args.camera_name)
    print("\nCalibration complete!")


if __name__ == "__main__":
    args = CliArgs().parse_args()
    main(args)
