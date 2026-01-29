from abc import ABC
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import cv2
import numpy as np
import numpy.typing as npt


@dataclass
class OpenCVBoard(ABC):
    pass

    def create_chessboard_points(self) -> np.ndarray:
        pass

    def detect_chessboard_corners(
        self, image: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray]]:
        pass

    def get_object_points(self, corners: np.ndarray) -> np.ndarray:
        pass


@dataclass
class OpenCVChessboard:
    corners_per_row: int
    corners_per_column: int
    square_size: float

    def __post_init__(self):
        self.partern_size = (self.corners_per_row, self.corners_per_column)

    def create_chessboard_points(self) -> np.ndarray:
        objp = np.zeros((self.corners_per_row * self.corners_per_column, 3), np.float32)

        objp[:, :2] = np.mgrid[
            0 : self.corners_per_row, 0 : self.corners_per_column
        ].T.reshape(-1, 2)  # RIGHT

        ## Kept here for reference
        # objp[:,:2] = np.mgrid[0:self.corners_per_column, 0:self.corners_per_row].T.reshape(-1,2) # WRONG

        return objp

    def detect_chessboard_corners(
        self, image: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.partern_size, None)
        return ret, corners


@dataclass
class OpenCVCharucoBoard:
    squares_in_x_dim: int
    squares_in_y_dim: int
    aruco_dict: Any
    square_length: float
    marker_lenght: float

    def __post_init__(self):
        self.board = cv2.aruco.CharucoBoard(
            (self.squares_in_x_dim, self.squares_in_y_dim),
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
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict)

        if len(corners) < 0:  # No aruco markers detected
            print("Error: No aruco markers detected.")
            return False, None, None

        # Interpolate Charuco corners
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners, markerIds=ids, image=gray, board=self.board
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
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        obj_points, img_points = self.board.matchImagePoints(
            charuco_corners,  # type: ignore
            charuco_ids,
        )

        obj_points = np.asarray(obj_points, dtype=np.float32)
        img_points = np.asarray(img_points, dtype=np.float32)

        return obj_points, img_points
