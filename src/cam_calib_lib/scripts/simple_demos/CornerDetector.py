## Camera calibration with a Charuco board with Opencv 4.9.0

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt


@dataclass
class CharucoBoard:
    row: int
    cols: int
    aruco_dict: Any
    square_length: float
    marker_lenght: float

    def __post_init__(self):
        self.board = cv2.aruco.CharucoBoard(
            (self.cols, self.row),
            squareLength=self.square_length,
            markerLength=self.marker_lenght,
            dictionary=self.aruco_dict,
        )

    def detect_charuco_corners(
        self, frame: npt.NDArray[np.uint8]
    ) -> Tuple[
        bool, Optional[npt.NDArray[np.float32]], Optional[npt.NDArray[np.int32]]
    ]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ## Step 1: ArUco markers detection.
        ## marker_corners --> tuple[npt.NDArray[np.float32]]
        ## ids --> npt.NDArray[np.int32] --> [N,]
        ## Four corners are detected for each marker. Each marker has an unique id.
        marker_corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict)

        if len(marker_corners) < 0:  # No aruco markers detected
            print("Error: No aruco markers detected.")
            return False, None, None

        ## Step 2: Identify Charuco board corners.
        ## charuco_corners --> npt.NDArray[np.float32] --> [N,1,2]
        ## charuco_ids --> npt.NDArray[np.int32] --> [N, 1]
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=marker_corners, markerIds=ids, image=gray, board=self.board
        )

        if retval < 3:  # minimum number of charuco corners
            print("Error: failed to interpolate Charuco corners.")
            return False, None, None

        # For type hints.
        charuco_corners = np.asarray(charuco_corners, dtype=np.float32)
        charuco_ids = np.asarray(charuco_ids, dtype=np.int32)

        return True, charuco_corners, charuco_ids

    def get_object_points(
        self,
        charuco_corners: npt.NDArray[np.float32],
        charuco_ids: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.float32]:

        obj_points, img_points = self.board.matchImagePoints(
            charuco_corners,  # type: ignore
            charuco_ids,
        )

        obj_points = np.asarray(obj_points, dtype=np.float32)

        ## Image points are the same charuco corners.
        assert np.all(
            np.isclose(charuco_corners, img_points)
        ), "Charuco corners and image points are not the same."
        # img_points = np.asarray(img_points, dtype=np.float32)

        return obj_points


def show_image(img):
    window_name = "window"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_charuco_corners(frame, board):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

    if len(corners) > 0:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Interpolate Charuco corners
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners, markerIds=ids, image=gray, board=board
        )

        if retval > 3:  # minimum number of charuco corners
            cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)

            obj_points, img_points = board.matchImagePoints(
                charuco_corners,
                charuco_ids,
            )
        else:
            print("Not enough corners detected.")

        show_image(frame)
        pass

    ## Calibration options:
    ## Option 1.
    # ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    #     charucoCorners=all_corners,
    #     charucoIds=all_ids,
    #     board=charuco_board,
    #     imageSize=image_size,
    #     cameraMatrix=None,
    #     distCoeffs=None
    # )

    ## Option 2.
    # ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


if __name__ == "__main__":
    from pyprojroot.here import here

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

    img_path = here() / "assets/sample_data_charuco/frame_7x10.png"
    assert img_path.exists(), f"Image not found in {img_path}. Please check the path."

    frame = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

    frame = np.asarray(frame, dtype=np.uint8)

    ret, charuco_corners, charuco_ids = charuco_board_7x10.detect_charuco_corners(frame)

    if (
        charuco_corners is None or charuco_ids is None or not ret
    ):  # Needed or type hint will complain.
        exit("Error: No Charuco corners detected.")

    obj_points = charuco_board_7x10.get_object_points(charuco_corners, charuco_ids)
    img_points = charuco_corners

    cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)  # type: ignore[no-untyped-call]

    # obj_points = np.asarray(obj_points, dtype=np.float32)
    # img_points = np.asarray(img_points, dtype=np.float32)

    print(f"Charuco corner[0]: {charuco_corners[0]}")
    print(f"Charuco id[0]: {charuco_ids[0]}")
    print(f"Object points[0]: {obj_points[0]}")
    print(f"Image points[0]: {img_points[0]}")

    cv2.imshow("Charuco Corners", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
