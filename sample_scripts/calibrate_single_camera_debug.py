from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tap import Tap


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
        ].T.reshape(
            -1, 2
        )  # RIGHT
        # objp[:,:2] = np.mgrid[0:self.corners_per_column, 0:self.corners_per_row].T.reshape(-1,2) # WRONG
        return objp


class CliArgs(Tap):
    images_path: Path
    chessboard_rows: int
    chessboard_cols: int
    square_size: float  # in meters
    camera: str  # "left" or "right"


def main(args: CliArgs):
    assert isinstance(args.images_path, Path)
    assert args.images_path.exists(), f"Images path does not exist: {args.images_path}"
    assert args.camera in [
        "left",
        "right",
    ], f"Camera must be 'left' or 'right', got '{args.camera}'"

    # Load images for the specified camera
    images = sorted(
        list(args.images_path.glob("*.png")) + list(args.images_path.glob("*.jpg"))
    )

    if len(images) == 0:
        raise ValueError(f"No valid images found for {args.camera} camera!")

    # Get image size from first image
    first_img = cv2.imread(str(images[0]))
    if first_img is None:
        raise ValueError(f"Could not load first image: {images[0]}")
    # image_size = (first_img.shape[1], first_img.shape[0])

    pattern_size = (args.chessboard_rows, args.chessboard_cols)
    print(f"Pattern size: {pattern_size}")

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    chessboard = OpenCVChessboard(
        corners_per_row=args.chessboard_rows,
        corners_per_column=args.chessboard_cols,
        square_size=args.square_size,
    )
    objp = chessboard.create_chessboard_points()

    # # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # total_obj_points = args.chessboard_rows * args.chessboard_cols
    # objp = np.zeros((total_obj_points,3), np.float32)

    # objp[:,:2] = np.mgrid[0:pattern_size[0],0:pattern_size[1]].T.reshape(-1,2)
    # # objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2) # WRONG
    # # objp[:,:2] = np.mgrid[0:6,0:7].T.reshape(-1,2) # RIGHT
    # print(f"Objp: {objp.shape}")

    # Arrays to store object points and image points from all the images.
    obj_points = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        # If found, add object points, image points (after refining them)
        if ret:
            obj_points.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
            cv2.namedWindow(f"Corners for image {fname.name}", cv2.WINDOW_NORMAL)
            cv2.imshow(f"Corners for image {fname.name}", img)
            cv2.waitKey(500)

        if ret:
            print(f"✔️ Corners found for image {fname.name}")
        else:
            print(f"❌ Corners not found for image {fname.name}")

    cv2.destroyAllWindows()

    # if ret:
    #     print(f"✔️ Corners found for image {fname.name}")
    # else:
    #     print(f"❌ Corners not found for image {fname.name}")

    if len(obj_points) < 10:
        print(
            f"Warning: Only {len(obj_points)} valid images found. At least 10 images are recommended for good calibration."
        )

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, imgpoints, gray.shape[::-1], None, None
    )

    print(f"Reprojection error: {ret}")

    ## Evaluate calibration
    first_img = cv2.imread(str(images[0]))

    if first_img is not None and len(obj_points) > 0:
        first_img = np.asarray(first_img, dtype=np.uint8)

        ## Display projected corners debug.
        i = 0
        print(obj_points[i].shape)
        imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        imgpoints2 = np.squeeze(imgpoints2)

        img_copy = first_img.copy()
        cv2.drawChessboardCorners(img_copy, pattern_size, imgpoints2, True)
        cv2.namedWindow(
            f"Projected corners for image {images[i].name}", cv2.WINDOW_NORMAL
        )
        cv2.imshow(f"Projected corners for image {images[i].name}", img_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("\nCalibration complete!")


if __name__ == "__main__":
    args = CliArgs().parse_args()
    main(args)
