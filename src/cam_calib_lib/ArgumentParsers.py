from pathlib import Path
import cv2
from tap import Tap

def aruco_dict_from_string_string_list(name: str) -> int:
    """
    Convert a string like 'DICT_4X4_250' to cv2.aruco.DICT_4X4_250
    """
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Invalid ArUco dictionary name: {name}")

    return getattr(cv2.aruco, name)

def cv2_flag_from_string(flag: str) -> int:
    """
    Convert a string like 'CALIB_FIX_K3' to cv2.CALIB_FIX_K3
    """
    if not hasattr(cv2, flag):
        raise ValueError(f"Invalid calibration flag: {flag}")

    return getattr(cv2, flag)

def compose_cv2_flags(flags: list[str]) -> int:
    """
    Convert a list of strings like ['CALIB_FIX_K3', 'CALIB_ZERO_TANGENT_DIST']
    into the corresponding cv2 bitmask using the | operator.
    """
    assert isinstance(flags, list), "flags must be a list of strings"

    if len(flags) == 0:
        return 0

    final_flag = 0
    for name in flags:
        final_flag |= cv2_flag_from_string(name)

    return final_flag

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

    # Opencv flags
    aruco_dict: str = "DICT_4X4_250"
    distortion_flags: list[str] = ["CALIB_FIX_K3", "CALIB_ZERO_TANGENT_DIST"]
    stereo_flags: list[str] = ["CALIB_FIX_INTRINSIC"]

    def configure(self):
        self.add_argument(
            "--distortion_flags",
            help="Extra configuration options for distortion model."
            + " Default: [CALIB_FIX_K3, CALIB_ZERO_TANGENT_DIST]",
            nargs="+",
        )
        self.add_argument(
            "--stereo_flags",
            help=("Extra configuration options for stereo calibration."
             +" Default: CALIB_FIX_INTRINSIC. For MockOR Microscope use: CALIB_USE_INTRINSIC_GUESS"),
            nargs="+",
        )
