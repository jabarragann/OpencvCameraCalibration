from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
import numpy.typing as npt
from cam_calib_lib.CornerDetector import CharucoBoard
from cam_calib_lib.Utils import save_calibration_yaml
from pyprojroot.here import here


def load_data(
    file_path: Path,
) -> Tuple[List[npt.NDArray[np.uint8]], List[npt.NDArray[np.uint8]]]:
    files = sorted(file_path.glob("*.png"))

    list_of_images = []
    for file in files:
        img = cv2.imread(str(file), cv2.IMREAD_COLOR)
        img = np.asarray(img, dtype=np.uint8)

        list_of_images.append(img)

    train = int(len(list_of_images) * 0.8)
    test = len(list_of_images) - train

    print(f"Loaded {len(files)} images from {file_path}")
    print(f"Training images: {train}, Testing images: {test}")

    train_images = list_of_images[:train]
    test_images = list_of_images[train:]

    return train_images, test_images


def get_all_object_points(
    list_of_images: List[npt.NDArray[np.uint8]], charuco_board: CharucoBoard
) -> Tuple[List[npt.NDArray[np.float32]], List[npt.NDArray[np.float32]]]:
    all_obj_poinst: List[npt.NDArray[np.float32]] = []
    all_img_points: List[npt.NDArray[np.float32]] = []

    for idx, frame in enumerate(list_of_images):
        ret, charuco_corners, charuco_ids = charuco_board.detect_charuco_corners(frame)

        if (
            charuco_corners is None or charuco_ids is None or not ret
        ):  # Needed or type hint will complain.
            print(f"No Charuco corners detected for image {idx}")
            continue

        obj_points, img_points = charuco_board.get_object_points(
            charuco_corners, charuco_ids
        )

        all_obj_poinst.append(obj_points)
        all_img_points.append(img_points)

    # obj_points_nda = np.concatenate(all_obj_poinst, axis=0)
    # img_points_nda = np.concatenate(all_img_points, axis=0)

    # return obj_points_nda, img_points_nda

    return all_obj_poinst, all_img_points


def calculate_reprojection_error(
    object_points: List[npt.NDArray[np.float32]],
    image_points: List[npt.NDArray[np.float32]],
    rvecs: List[npt.NDArray[np.float32]],
    tvecs: List[npt.NDArray[np.float32]],
    mtx: npt.NDArray[np.float32],
    dist: npt.NDArray[np.float32],
) -> float:
    mean_error = 0
    for i in range(len(object_points)):
        imgpoints2, _ = cv2.projectPoints(  ## Assuming distortion
            object_points[i], rvecs[i], tvecs[i], mtx, dist
        )

        # imgpoints2, _ = cv2.projectPoints( ## Assuming no distortion
        #     object_points[i], rvecs[i], tvecs[i], mtx, np.zeros_like(dist)
        # )
        error = cv2.norm(image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    mean_error /= len(object_points)

    # print(f"total error: {mean_error}")

    return mean_error


def main():
    # Board parameters
    SQUARE_LENGTH = 0.0080  # in meters
    MARKER_LENGTH = 0.0048  # in meters
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    CHARUCO_ROWS = 7
    CHARUCO_COLS = 10

    charuco_board_7x10 = CharucoBoard(
        row=CHARUCO_ROWS,
        cols=CHARUCO_COLS,
        aruco_dict=aruco_dict,
        square_length=SQUARE_LENGTH,
        marker_lenght=MARKER_LENGTH,
    )

    root_path = here()  # Path to root path from pyprojroot.
    img_path = root_path / "output/frames_5fps"
    train_images, test_images = load_data(img_path)

    object_points, image_points = get_all_object_points(
        train_images, charuco_board_7x10
    )

    height, width = train_images[0].shape[:2]
    image_size = (width, height)

    # print(f"Object Points Shape: {object_points.shape}")
    # print(f"Image Points Shape: {image_points.shape}")

    ########################
    ## Undistortion model flags - Setting the right model is important for correct undistortion.

    ## No flags lead to weird warpping after undistortion - the problem is k_3 coefficient
    # flags = 0

    ## Rational model improve results but overcomplicates the model.
    ## Rational model enables 6 coefficients for the radial distortion.
    # flags = cv2.CALIB_RATIONAL_MODEL ## This is important to correctly undistort the images

    ## BEST fix - Set K3 to zero
    flags = cv2.CALIB_FIX_K3

    ## Setting tangent to zero didn't change much the results
    # flags = cv2.CALIB_ZERO_TANGENT_DIST
    ########################

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points,  # type: ignore
        image_points,
        image_size,
        None,  # type: ignore
        None,  # type: ignore
        flags=flags,
    )

    print(f"Camera Matrix:\n{mtx}")
    print(f"Distortion Coefficients:\n{dist}")
    # print(f"Rotation Vectors:\n{rvecs}")
    # print(f"Translation Vectors:\n{tvecs}")

    save_calibration_yaml(
        filename=str(img_path / "camera_calibration.yaml"),
        image_size=image_size,
        mtx=mtx,
        dist=dist,
        camera_name="charuco_camera",
    )

    ## Reprojection error
    train_mean_error = calculate_reprojection_error(
        object_points, image_points, rvecs, tvecs, mtx, dist
    )
    print(f"Reprojection error - Train images: {len(train_images)}")
    print(f"Mean error: {train_mean_error:.4f} pixels")

    ## Reprojection on test images
    test_obj_points, test_img_points = get_all_object_points(
        test_images, charuco_board_7x10
    )
    test_rvecs = []
    test_tvecs = []
    for idx, frame in enumerate(test_images):
        success, rvec, tvec = cv2.solvePnP(
            test_obj_points[idx],
            test_img_points[idx],
            mtx,
            dist,
        )
        test_rvecs.append(rvec)
        test_tvecs.append(tvec)

    test_mean_error = calculate_reprojection_error(
        test_obj_points, test_img_points, test_rvecs, test_tvecs, mtx, dist
    )
    print(f"Reprojection error - Test images: {len(test_images)}")
    print(f"Mean error: {test_mean_error:.4f} pixels")


if __name__ == "__main__":
    main()
