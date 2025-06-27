from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
import numpy.typing as npt
from cam_calib_lib.CornerDetector import CharucoBoard


def load_data(file_path: Path) -> List[npt.NDArray[np.uint8]]:
    files = sorted(file_path.glob("*.png"))

    list_of_images = []
    for file in files:
        img = cv2.imread(str(file), cv2.IMREAD_COLOR)
        img = np.asarray(img, dtype=np.uint8)

        list_of_images.append(img)

    print(f"Loaded {len(files)} images from {file_path}")
    return list_of_images


def get_all_object_points(
    list_of_images: List[npt.NDArray[np.uint8]], charuco_board: CharucoBoard
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
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

    img_path = Path(__file__).parent / "../../output/frames_5fps"
    list_of_images = load_data(img_path)

    object_points, image_points = get_all_object_points(
        list_of_images, charuco_board_7x10
    )

    height, width = list_of_images[0].shape[:2]
    image_size = (width, height)

    # print(f"Object Points Shape: {object_points.shape}")
    # print(f"Image Points Shape: {image_points.shape}")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )

    print(f"Camera Matrix:\n{mtx}")
    print(f"Distortion Coefficients:\n{dist}")
    # print(f"Rotation Vectors:\n{rvecs}")
    # print(f"Translation Vectors:\n{tvecs}")


if __name__ == "__main__":
    main()
