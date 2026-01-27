from pathlib import Path
from tap import Tap


class CameraCalibrationArgs(Tap):
    images_path: Path
    output_path: Path
    camera_name: str
    corners_per_board_row: int
    corners_per_board_column: int
    square_size: float  # in meters