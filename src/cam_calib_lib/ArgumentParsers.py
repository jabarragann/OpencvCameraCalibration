from pathlib import Path
import cv2
from tap import Tap


class CameraCalibrationArgs(Tap):
    images_path: Path
    output_path: Path
    camera_name: str
    corners_per_board_row: int
    corners_per_board_column: int
    square_size: float  # in meters

class CameraCalibrationCharucoArgs(Tap):
    images_path: Path
    output_path: Path
    camera_name: str
    squares_in_x_dim: int
    squares_in_y_dim: int
    square_size: float  # in meters
    marker_size: float  # in meters
    aruco_dict: int = cv2.aruco.DICT_4X4_250
    flags: int = cv2.CALIB_FIX_K3