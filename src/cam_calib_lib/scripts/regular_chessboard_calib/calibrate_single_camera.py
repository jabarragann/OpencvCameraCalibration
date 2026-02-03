"""
Single camera calibration using regular chessboard pattern.
Performs calibration for either left or right camera.
"""

import re
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
import numpy.typing as npt

from cam_calib_lib.ArgumentParsers import CameraCalibrationArgs
from cam_calib_lib.OpencvBoards import OpenCVChessboard
from cam_calib_lib.Utils import save_calibration_yaml_single_cam


class CliArgs(CameraCalibrationArgs):
    print_debug: bool = False
    flags: int = cv2.CALIB_FIX_K3


def load_camera_images(images_path: Path, camera: str) -> List[Path]:
    """
    Load images for the specified camera from the folder.
    Expected naming format: left-xxxx.png or right-xxxx.png
    """
    images: List[Path] = []

    # Find all images for the specified camera
    all_files = sorted(
        list(images_path.glob("*.png")) + list(images_path.glob("*.jpg"))
    )

    # Extract frame numbers from filenames
    image_dict = {}

    prefix = f"{camera}-"
    for file in all_files:
        filename = file.name
        if filename.startswith(prefix):
            match = re.search(rf"{camera}-(\d+)\.(png|jpg)", filename)
            if match:
                frame_num = int(match.group(1))
                image_dict[frame_num] = file

    # Sort by frame number
    sorted_frames = sorted(image_dict.keys())
    for frame_num in sorted_frames:
        images.append(image_dict[frame_num])

    print(f"Found {len(images)} images for {camera} camera")
    return images


def prepare_calibration_data(
    regular_chessboard: OpenCVChessboard,
    images: List[Path],
    print_debug: bool = False,
) -> Tuple[
    List[npt.NDArray[np.float32]],
    List[npt.NDArray[np.float32]],
]:
    """
    Prepare object points and image points for single camera calibration.

    Returns:
        (obj_points, img_points)
    """

    objp = regular_chessboard.get_object_points(None, None)  # type: ignore

    obj_points: List[npt.NDArray[np.float32]] = []
    img_points: List[npt.NDArray[np.float32]] = []

    print("Detecting chessboard corners...")
    for idx, img_path in enumerate(images):
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"Warning: Could not load image {idx}: {img_path}")
            continue

        img = np.asarray(img, dtype=np.uint8)

        # Detect corners
        ret, corners = regular_chessboard.detect_corners(img)
        if ret:
            obj_points.append(objp)
            img_points.append(corners)

        # if ret:
        #     img_copy = img.copy()
        #     cv2.namedWindow(f"Corners for image {img_path.name}", cv2.WINDOW_NORMAL)
        #     cv2.drawChessboardCorners(img_copy, pattern_size, corners, ret)
        #     cv2.imshow(f"Corners for image {img_path.name}", img_copy)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        if print_debug:
            if ret:
                print(f"  Image {idx}: ✓ Corners detected")
            else:
                print(f"  Image {idx}: ✗ Failed to detect corners")

    print(f"\nSuccessfully detected corners in {len(obj_points)}/{len(images)} images")
    return obj_points, img_points


def calculate_reprojection_error(
    obj_points: List[npt.NDArray[np.float32]],
    img_points: List[npt.NDArray[np.float32]],
    rvecs: List[npt.NDArray[np.float32]],
    tvecs: List[npt.NDArray[np.float32]],
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


def perform_single_calibration(
    obj_points: List[npt.NDArray[np.float32]],
    img_points: List[npt.NDArray[np.float32]],
    image_size: Tuple[int, int],
    flags: int = cv2.CALIB_FIX_K3,
) -> Tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    List[npt.NDArray[np.float32]],
    List[npt.NDArray[np.float32]],
    float,
]:
    """
    Perform single camera calibration.

    Returns:
        (camera_matrix, dist_coeffs, rvecs, tvecs, reprojection_error)
    """
    print("\nPerforming camera calibration...")

    # flags = (
    #     cv2.CALIB_FIX_K1
    #     | cv2.CALIB_FIX_K2
    #     | cv2.CALIB_FIX_K3
    #     | cv2.CALIB_FIX_K4
    #     | cv2.CALIB_FIX_K5
    #     | cv2.CALIB_FIX_K6
    #     | cv2.CALIB_ZERO_TANGENT_DIST
    # )

    #     flags = (
    #     cv2.CALIB_FIX_K3 |
    #     cv2.CALIB_FIX_K4 |
    #     cv2.CALIB_FIX_K5 |
    #     cv2.CALIB_FIX_K6 |
    #     cv2.CALIB_ZERO_TANGENT_DIST
    # )

    flags = cv2.CALIB_FIX_K3

    # flags = None

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points,
        img_points,
        image_size,
        None,
        None,
        flags=flags,
    )

    print("\nCamera calibration completed!")
    print(f"Reprojection error (from calibration): {ret:.4f} pixels")
    print(f"\nCamera matrix:\n{camera_matrix}")
    print(f"\nDistortion coefficients:\n{dist_coeffs}")

    return camera_matrix, dist_coeffs, rvecs, tvecs, ret


def display_projected_corners(
    img: npt.NDArray[np.uint8],
    obj_points: npt.NDArray[np.float32],
    ground_truth_corners: npt.NDArray[np.float32],
    rvec: npt.NDArray[np.float32],
    tvec: npt.NDArray[np.float32],
    camera_matrix: npt.NDArray[np.float32],
    dist_coeffs: npt.NDArray[np.float32],
    camera: str,
    use_distortion: bool = True,
):
    """
    Display projected corners and ground truth corners on the image.

    Args:
        img: Input image
        obj_points: 3D object points
        ground_truth_corners: Detected corners from the image (ground truth)
        rvec: Rotation vector for this image
        tvec: Translation vector for this image
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        camera: Camera name (for window title)
        use_distortion: If True, use distortion model. If False, ignore distortion.
    """
    # Project 3D points to 2D image points
    dist_to_use = dist_coeffs if use_distortion else np.zeros_like(dist_coeffs)
    projected_points, _ = cv2.projectPoints(
        obj_points, rvec, tvec, camera_matrix, dist_to_use
    )
    projected_points = projected_points.reshape(-1, 2)

    # Reshape ground truth corners to (N, 2) format
    ground_truth_2d = ground_truth_corners.reshape(-1, 2)

    # Create a copy of the image for drawing
    img_display = img.copy()

    # Draw ground truth corners in green
    for pt in ground_truth_2d:
        pt_int = tuple(map(int, pt))
        cv2.circle(img_display, pt_int, 5, (0, 255, 0), -1)  # Green filled circle

    # Draw projected corners in red
    for pt in projected_points:
        pt_int = tuple(map(int, pt))
        cv2.circle(img_display, pt_int, 5, (0, 0, 255), -1)  # Red filled circle

    # Draw lines connecting corresponding points to visualize error
    for gt_pt, proj_pt in zip(ground_truth_2d, projected_points):
        gt_pt_int = tuple(map(int, gt_pt))
        proj_pt_int = tuple(map(int, proj_pt))
        cv2.line(img_display, gt_pt_int, proj_pt_int, (255, 255, 0), 1)  # Yellow line

    # Calculate and display mean error for this image
    error = cv2.norm(ground_truth_2d, projected_points, cv2.NORM_L2) / len(
        projected_points
    )

    # Add text with error information
    distortion_text = "with distortion" if use_distortion else "without distortion"
    text = f"Reprojection error ({distortion_text}): {error:.4f} pixels"
    cv2.putText(
        img_display,
        text,
        (10, 400),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (150, 150, 0),
        2,
    )
    cv2.putText(
        img_display,
        "Green: Ground truth, Red: Projected, Yellow: Error vectors",
        (10, 430),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (150, 150, 0),
        2,
    )

    # Display the image
    window_name = f"Projected vs Ground Truth Corners ({camera.capitalize()})"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img_display)
    print("Showing projected vs ground truth corners. Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def display_undistorted_image(
    img: npt.NDArray[np.uint8],
    mtx: npt.NDArray[np.float32],
    dist: npt.NDArray[np.float32],
    camera: str,
):
    """
    Display the undistorted image using the camera intrinsics.
    """
    w, h = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    # undistorted = cv2.undistort(img, mtx, dist)
    undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)

    #### debug
    new_camera_matrix_alpha0, _ = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 0, (w, h)
    )
    new_camera_matrix_alpha1, _ = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 1, (w, h)
    )

    def pretty_matrix(mat, title):
        np.set_printoptions(precision=2, suppress=True, floatmode="fixed")
        print(f"{title}:\n{mat}\n")

    pretty_matrix(mtx, "Original Camera Matrix (mtx)")
    pretty_matrix(new_camera_matrix_alpha0, "New Camera Matrix (alpha=0)")
    pretty_matrix(new_camera_matrix_alpha1, "New Camera Matrix (alpha=1)")
    ####

    concatenated = np.hstack([img, undistorted])
    cv2.namedWindow(
        f"Original/Undistorted {camera.capitalize()} Image", cv2.WINDOW_NORMAL
    )

    cv2.imshow(f"Original/Undistorted {camera.capitalize()} Image", concatenated)
    print(f"Showing original/undistorted {camera} image. Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_calibration_results(
    output_path: Path,
    camera_matrix: npt.NDArray[np.float32],
    dist_coeffs: npt.NDArray[np.float32],
    image_size: Tuple[int, int],
    reprojection_error: float,
    camera: str,
) -> None:
    """Save calibration results to numpy files in the same format as stereo calibration."""
    output_path.mkdir(parents=True, exist_ok=True)

    save_calibration_yaml_single_cam(
        filename=str(output_path / f"{camera}_calibration.yaml"),
        image_size=image_size,
        mtx=camera_matrix,
        dist=dist_coeffs,
        camera_name=camera,
    )

    # # Save in the same format as stereo calibration
    # np.save(str(output_path / f"{camera}_camera_matrix.npy"), camera_matrix)
    # np.save(str(output_path / f"{camera}_dist_coeffs.npy"), dist_coeffs)

    # # Save metadata
    # metadata = {
    #     "image_size": image_size,
    #     "reprojection_error": reprojection_error,
    # }
    # np.save(str(output_path / "metadata.npy"), metadata)

    # print(f"\nCalibration results saved to {output_path}")


def main(args: CliArgs):
    assert isinstance(args.images_path, Path)
    assert isinstance(args.output_path, Path)
    assert args.images_path.exists(), f"Images path does not exist: {args.images_path}"
    assert args.camera_name in ["left", "right"], (
        f"Camera must be 'left' or 'right', got '{args.camera_name}'"
    )

    regular_chessboard = OpenCVChessboard(
        corners_per_row=args.corners_per_board_row,
        corners_per_column=args.corners_per_board_column,
        square_size=args.square_size,
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
        regular_chessboard, images, print_debug=args.print_debug
    )

    if len(obj_points) < 10:
        print(
            f"Warning: Only {len(obj_points)} valid images found. At least 10 images are recommended for good calibration."
        )

    # Perform single camera calibration
    camera_matrix, dist_coeffs, rvecs, tvecs, reprojection_error = (
        perform_single_calibration(obj_points, img_points, image_size, args.flags)
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
