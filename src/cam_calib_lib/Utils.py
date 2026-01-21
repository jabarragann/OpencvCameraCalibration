import yaml
import numpy as np

import yaml
import numpy as np


def load_camera_yaml(filename: str):
    with open(filename, "r") as f:
        calib_data = yaml.safe_load(f)

    image_width = calib_data["image_width"]
    image_height = calib_data["image_height"]
    camera_name = calib_data.get("camera_name", "unknown")

    camera_matrix = np.array(
        calib_data["camera_matrix"]["data"], dtype=np.float64
    ).reshape((3, 3))
    distortion_coeffs = np.array(
        calib_data["distortion_coefficients"]["data"], dtype=np.float64
    )

    # rectification_matrix = np.array(
    #     calib_data["rectification_matrix"]["data"], dtype=np.float64
    # ).reshape((3, 3))
    # projection_matrix = np.array(
    #     calib_data["projection_matrix"]["data"], dtype=np.float64
    # ).reshape((3, 4))

    return {
        "image_size": (image_width, image_height),
        "camera_name": camera_name,
        "camera_matrix": camera_matrix,
        "dist_coeffs": distortion_coeffs,
        # "rectification_matrix": rectification_matrix,
        # "projection_matrix": projection_matrix,
    }


def save_camera_yaml(
    filename: str,
    image_size: tuple[int, int],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    camera_name: str = "my_camera",
):
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    data = {
        "image_width": image_size[0],
        "image_height": image_size[1],
        "camera_name": camera_name,
        "camera_matrix": {
            "rows": 3,
            "cols": 3,
            "data": camera_matrix.flatten(order="C").tolist(),
        },
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {
            "rows": 1,
            "cols": len(dist_coeffs.flatten()),
            "data": dist_coeffs.flatten().tolist(),
        },
        # "rectification_matrix": {
        #     "rows": 3,
        #     "cols": 3,
        #     "data": np.eye(3, dtype=float).flatten().tolist()
        # },
        # "projection_matrix": {
        #     "rows": 3,
        #     "cols": 4,
        #     "data": [
        #         fx, 0.0, cx, 0.0,
        #         0.0, fy, cy, 0.0,
        #         0.0, 0.0, 1.0, 0.0
        #     ]
        # }
    }

    print(f"Saving camera calibration to {filename}...")
    with open(filename, "w") as f:
        yaml.dump(data, f, sort_keys=False)
