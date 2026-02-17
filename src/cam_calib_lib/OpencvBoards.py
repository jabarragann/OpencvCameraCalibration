from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import cv2
import numpy as np
import numpy.typing as npt


@dataclass
class OpenCVCornerDetector(ABC):
    @abstractmethod
    def detect_corners(
        self, image: npt.NDArray[np.uint8]
    ) -> Tuple[bool, Optional[npt.NDArray[np.float32]], Optional[npt.NDArray[np.int32]]]:
        # return bool, corners, corners_ids

        pass

    @abstractmethod
    def get_object_points(
        self, corners: npt.NDArray[np.float32], ids: npt.NDArray[np.int32]
    ) -> npt.NDArray[np.float32]:
        pass


@dataclass
class OpenCVChessboard(OpenCVCornerDetector):
    corners_per_row: int
    corners_per_column: int
    square_size: float

    def __post_init__(self):
        self.pattern_size = (self.corners_per_row, self.corners_per_column)
        self.obj_points = self.create_chessboard_points()

    def create_chessboard_points(self) -> np.ndarray:
        objp = np.zeros((self.corners_per_row * self.corners_per_column, 3), np.float32)

        objp[:, :2] = np.mgrid[
            0 : self.corners_per_row, 0 : self.corners_per_column
        ].T.reshape(-1, 2)  # RIGHT

        objp *= self.square_size

        ## Kept here for reference
        # object points generation assumes x direction is along columns and y direction is along rows
        # objp[:,:2] = np.mgrid[0:self.corners_per_column, 0:self.corners_per_row].T.reshape(-1,2) # WRONG

        return objp

    def detect_corners(
        self, image: npt.NDArray[np.uint8]
    ) -> Tuple[
        bool, Optional[npt.NDArray[np.float32]], Optional[npt.NDArray[np.int32]]
    ]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(
            gray,
            self.pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK
            + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            corners_ids = None

            corners_f32: npt.NDArray[np.float32] = np.asarray(
                corners, dtype=np.float32
            )
            return True, corners_f32, corners_ids 

        return False, None, None

    def get_object_points(
        self, corners: npt.NDArray[np.float32], ids: npt.NDArray[np.int32]
    ) -> npt.NDArray[np.float32]:
        # parameters 'corners' and 'ids' are kept to keep a common interface with CharucoBoard
        return self.obj_points


@dataclass
class OpenCVCharucoBoard(OpenCVCornerDetector):
    squares_in_x_dim: int
    squares_in_y_dim: int
    aruco_dict: Any
    square_length: float
    marker_length: float

    def __post_init__(self):
        self.board = cv2.aruco.CharucoBoard(
            (self.squares_in_x_dim, self.squares_in_y_dim),
            squareLength=self.square_length,
            markerLength=self.marker_length,
            dictionary=self.aruco_dict,
        )

    # def detect_corners(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
    def detect_corners(
        self, image: npt.NDArray[np.uint8]
    ) -> Tuple[
        bool, Optional[npt.NDArray[np.float32]], Optional[npt.NDArray[np.int32]]
    ]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict)

        if len(corners) == 0:  # No aruco markers detected
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
        charuco_corners_f32: npt.NDArray[np.float32] = np.asarray(
            charuco_corners, dtype=np.float32
        )
        charuco_ids_i32: npt.NDArray[np.int32] = np.asarray(charuco_ids, dtype=np.int32)

        return True, charuco_corners_f32, charuco_ids_i32

    def get_object_points(
        self,
        corners: npt.NDArray[np.float32],
        corners_ids: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.float32]:

        obj_points, img_points = self.board.matchImagePoints(
            corners,  # type: ignore
            corners_ids,
        )

        obj_points = np.asarray(obj_points, dtype=np.float32)
        # img_points = np.asarray(img_points, dtype=np.float32)

        return obj_points


if __name__ == "__main__":
    chessboard = OpenCVChessboard(
        corners_per_row=7, corners_per_column=10, square_size=0.008
    )
    charuco_board = OpenCVCharucoBoard(
        squares_in_x_dim=7,
        squares_in_y_dim=10,
        aruco_dict=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250),
        square_length=0.008,
        marker_length=0.0048,
    )

    print("finish")
