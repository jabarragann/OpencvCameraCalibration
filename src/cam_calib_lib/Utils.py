from pathlib import Path
import yaml
import numpy as np
import cv2
import numpy.typing as npt


def index_to_bgr(idx: int, total: int) -> tuple[int, int, int]:
    """
    Deterministic, high-contrast color from index.
    """
    hue = int(180 * idx / max(total, 1))
    hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def load_camera_calib(filename: Path):

    assert isinstance(filename, Path), "Filename must be a Path object"
    assert filename.suffix == ".yaml", "Filename must be a YAML file"
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

    if "rectification_matrix" in calib_data:
        rectification_matrix = np.array(
            calib_data["rectification_matrix"]["data"], dtype=np.float64
        ).reshape((3, 3))
    else:
        rectification_matrix = None

    if "projection_matrix" in calib_data:
        projection_matrix = np.array(
            calib_data["projection_matrix"]["data"], dtype=np.float64
        ).reshape((3, 4))
    else:
        projection_matrix = None

    return {
        "image_size": (image_width, image_height),
        "camera_name": camera_name,
        "camera_matrix": camera_matrix,
        "dist_coeffs": distortion_coeffs,
        "rectification_matrix": rectification_matrix,
        "projection_matrix": projection_matrix,
    }


def load_stereo_camera_calib(filename: Path):
    assert isinstance(filename, Path), "Filename must be a Path object"
    assert filename.suffix == ".yaml", "Filename must be a YAML file"
    with open(filename, "r") as f:
        calib_data = yaml.safe_load(f)

    image_width = calib_data["image_width"]
    image_height = calib_data["image_height"]
    fundamental_matrix = np.array(
        calib_data["Fundamental"]["data"], dtype=np.float64
    ).reshape((3, 3))
    essential_matrix = np.array(
        calib_data["Essential"]["data"], dtype=np.float64
    ).reshape((3, 3))
    T = np.array(calib_data["T"]["data"], dtype=np.float64).reshape((3, 1))
    R = np.array(calib_data["R"]["data"], dtype=np.float64).reshape((3, 3))
    left_camera_matrix = np.array(
        calib_data["left_camera_matrix"]["data"], dtype=np.float64
    ).reshape((3, 3))
    left_distortion_coefficients = np.array(
        calib_data["left_distortion_coefficients"]["data"], dtype=np.float64
    ).reshape((1, 5))
    right_camera_matrix = np.array(
        calib_data["right_camera_matrix"]["data"], dtype=np.float64
    ).reshape((3, 3))
    right_distortion_coefficients = np.array(
        calib_data["right_distortion_coefficients"]["data"], dtype=np.float64
    ).reshape((1, 5))

    return {
        "image_size": (image_width, image_height),
        "fundamental_matrix": fundamental_matrix,
        "essential_matrix": essential_matrix,
        "T": T,
        "R": R,
        "left_camera_matrix": left_camera_matrix,
        "left_distortion_coefficients": left_distortion_coefficients,
        "right_camera_matrix": right_camera_matrix,
        "right_distortion_coefficients": right_distortion_coefficients,
    }


def format_array(arr: npt.NDArray[np.float64], per_row: int = 3) -> list[str]:
    values = arr.flatten(order="C")
    lines: list[str] = []

    for i in range(0, len(values), per_row):
        chunk = values[i : i + per_row]
        # Format each number: zero -> 0.0, otherwise 6 decimal points
        formatted: list[str] = []
        for v in chunk:
            if v == 0.0:
                s = f"{v:.1f}"
            else:
                s = f"{v:.6f}"

            # Split into integer and fractional parts
            if "." in s:
                int_part, frac_part = s.split(".")
                # Pad integer part to the left, fractional part to the right
                s = f"{int_part:>6}.{frac_part:<6}"
            else:
                s = f"{s:>12}"

            formatted.append(s)
        line = ", ".join(formatted)
        line += ","  # to the last element
        lines.append(line)

    return lines


def modify_string_for_yaml(formated_arry: list[str]):
    """
    Modifies formated array to fit into the yaml string.
    """

    first_line_text = "  data: ["
    other_line_text = "         "
    last_line_text_end = "]"

    assert len(formated_arry) == 3, "Expected 3 lines of data"
    results: list[str] = []
    results.append(first_line_text + formated_arry[0])
    results.append(other_line_text + formated_arry[1])
    # remove the last comma from the last line
    results.append(other_line_text + formated_arry[2][:-1] + last_line_text_end)

    return "\n".join(results)


def create_yaml_str_single_cam(
    image_size: tuple[int, int], mtx: npt.NDArray[np.float64], dist: npt.NDArray[np.float64], camera_name: str
):
    mtx_str_data = modify_string_for_yaml(format_array(mtx))

    return f"""image_width: {image_size[0]}
image_height: {image_size[1]}
camera_name: {camera_name}
camera_matrix:
  rows: 3
  cols: 3
{mtx_str_data}
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: {dist.size}
  data: [{", ".join(f"{v:.6f}" for v in dist.flatten())}]
"""


def create_yaml_str_stereo_cam(
    image_size: tuple[int, int],
    mtx: npt.NDArray[np.float64],
    dist: npt.NDArray[np.float64],
    camera_name: str,
    rect_mtx: npt.NDArray[np.float64],
    proj_mtx: npt.NDArray[np.float64],
):
    mtx_str_data = modify_string_for_yaml(format_array(mtx))
    rect_mtx_str_data = modify_string_for_yaml(format_array(rect_mtx))
    proj_mtx_str_data = modify_string_for_yaml(format_array(proj_mtx, per_row=4))

    return f"""image_width: {image_size[0]}
image_height: {image_size[1]}
camera_name: {camera_name}
camera_matrix:
  rows: 3
  cols: 3
{mtx_str_data}
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: {dist.size}
  data: [{", ".join(f"{v:.6f}" for v in dist.flatten())}]
rectification_matrix:
  rows: 3
  cols: 3
{rect_mtx_str_data}
projection_matrix:
  rows: 3
  cols: 4
{proj_mtx_str_data}
"""


def create_yaml_str_stereo_cam_additional(
    image_size: tuple[int, int],
    F: npt.NDArray[np.float64],
    E: npt.NDArray[np.float64],
    T: npt.NDArray[np.float64],
    R: npt.NDArray[np.float64],
    left_mtx: npt.NDArray[np.float64],
    left_dist: npt.NDArray[np.float64],
    right_mtx: npt.NDArray[np.float64],
    right_dist: npt.NDArray[np.float64],
):
    F_str_data = modify_string_for_yaml(format_array(F))
    E_str_data = modify_string_for_yaml(format_array(E))
    R_str_data = modify_string_for_yaml(format_array(R))
    left_mtx_str_data = modify_string_for_yaml(format_array(left_mtx))
    right_mtx_str_data = modify_string_for_yaml(format_array(right_mtx))

    return f"""image_width: {image_size[0]}
image_height: {image_size[1]}
Fundamental:
  rows: 3
  cols: 3
{F_str_data}
Essential:
  rows: 3
  cols: 3
{E_str_data}
T:
  rows: 1
  cols: {T.shape[0]}
  data: [{", ".join(f"{v:.6f}" for v in T.flatten())}]
R:
  rows: 3
  cols: 3
{R_str_data}
left_camera_matrix:
  rows: 3
  cols: 3
{left_mtx_str_data}
left_distortion_coefficients:
  rows: 1
  cols: {left_dist.size}
  data: [{", ".join(f"{v:.6f}" for v in left_dist.flatten())}]
right_camera_matrix:
  rows: 3
  cols: 3
{right_mtx_str_data}
right_distortion_coefficients:
  rows: 1
  cols: {right_dist.size}
  data: [{", ".join(f"{v:.6f}" for v in right_dist.flatten())}]
"""


def save_calibration_yaml_single_cam(
    filename: str,
    image_size: tuple[int, int],
    mtx: npt.NDArray[np.float64],
    dist: npt.NDArray[np.float64],
    camera_name: str,
):
    with open(filename, "w") as f:
        yaml_string = create_yaml_str_single_cam(image_size, mtx, dist, camera_name)
        f.write(yaml_string)


def save_calibration_yaml_stereo_cam(
    filename: str,
    image_size: tuple[int, int],
    mtx: npt.NDArray[np.float64],
    dist: npt.NDArray[np.float64],
    camera_name: str,
    rect_mtx: npt.NDArray[np.float64],
    proj_mtx: npt.NDArray[np.float64],
):
    with open(filename, "w") as f:
        yaml_string = create_yaml_str_stereo_cam(
            image_size, mtx, dist, camera_name, rect_mtx, proj_mtx
        )
        f.write(yaml_string)


def save_calibration_yaml_stereo_cam_additional(
    filename: str,
    image_size: tuple[int, int],
    F: npt.NDArray[np.float64],
    E: npt.NDArray[np.float64],
    T: npt.NDArray[np.float64],
    R: npt.NDArray[np.float64],
    left_mtx: npt.NDArray[np.float64],
    left_dist: npt.NDArray[np.float64],
    right_mtx: npt.NDArray[np.float64],
    right_dist: npt.NDArray[np.float64],
):
    with open(filename, "w") as f:
        yaml_string = create_yaml_str_stereo_cam_additional(
            image_size, F, E, T, R, left_mtx, left_dist, right_mtx, right_dist
        )
        f.write(yaml_string)


def save_calibration_parameters_regular(
    filename: Path,
    corners_per_row: int,
    corners_per_column: int,
    square_size: float,
    distortion_flags: list[str],
    stereo_calib_flags: list[str],
):
    params_dict: dict[str, int | float | list[str]] = {}
    params_dict["corners_per_row"] = corners_per_row
    params_dict["corners_per_column"] = corners_per_column
    params_dict["square_size"] = square_size
    params_dict["distortion_flags"] = distortion_flags
    params_dict["stereo_calib_flags"] = stereo_calib_flags

    with open(filename, "w") as f:
        yaml.dump(params_dict, f)

def save_calibration_parameters_charuco(
    filename: Path,
    aruco_dict: str,
    squares_per_row: int,
    squares_per_col: int,
    square_size: float,
    marker_size: float,
    distortion_flags: list[str],
    stereo_calib_flags: list[str],
):
    params_dict: dict[str, str | int | float | list[str]] = {}
    params_dict["aruco_dict"] = aruco_dict
    params_dict["squares_per_row"] = squares_per_row
    params_dict["squares_per_col"] = squares_per_col
    params_dict["square_size"] = square_size
    params_dict["marker_size"] = marker_size
    params_dict["distortion_flags"] = distortion_flags
    params_dict["stereo_calib_flags"] = stereo_calib_flags

    with open(filename, "w") as f:
        yaml.dump(params_dict, f)


if __name__ == "__main__":
    mtx = np.array(
        [
            [1794.471601, 0.0, 1043.039610],
            [0.0, 1806.394272, 479.158587],
            [0.0, 0.0, 1.000000],
        ]
    )
    dist = np.array([-0.323355, 0.464083, 0.001455, -0.008386, 0.000000])

    print("format_array")
    print(",\n".join(format_array(mtx)))
    print("modify_string_for_yaml")
    print(modify_string_for_yaml(format_array(mtx)))
    print("create_yaml_string")
    print(create_yaml_str_single_cam((1920, 1080), mtx, dist, "test"))

    print(format_array(mtx))
