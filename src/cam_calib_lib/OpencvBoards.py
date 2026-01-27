from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np

@dataclass
class OpenCVChessboard:
    corners_per_row: int
    corners_per_column: int
    square_size: float

    def __post_init__(self):
        self.partern_size = (self.corners_per_row, self.corners_per_column)

    def create_chessboard_points(self) -> np.ndarray:
        objp = np.zeros((self.corners_per_row * self.corners_per_column, 3), np.float32)

        objp[:,:2] = np.mgrid[0:self.corners_per_row, 0:self.corners_per_column].T.reshape(-1,2) # RIGHT

        ## Kept here for reference
        # objp[:,:2] = np.mgrid[0:self.corners_per_column, 0:self.corners_per_row].T.reshape(-1,2) # WRONG

        return objp
    
    def detect_chessboard_corners(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.partern_size, None)
        return ret, corners