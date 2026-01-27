from pathlib import Path
import yaml
import numpy as np


def load_camera_calib(filename: Path):
    print(type(filename))
    print(filename)

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

def format_array(arr, per_row=3):
    values = arr.flatten(order="C")
    lines = []

    for i in range(0, len(values), per_row):
        chunk = values[i : i + per_row]
        # Format each number: zero -> 0.0, otherwise 6 decimal points
        formatted = []
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
        line += "," # to the last element 
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
    results = []
    results.append(first_line_text + formated_arry[0])
    results.append(other_line_text + formated_arry[1])
    # remove the last comma from the last line 
    results.append(other_line_text + formated_arry[2][:-1] + last_line_text_end)

    return "\n".join(results)


def create_yaml_string(
    image_size: tuple[int, int], mtx: np.ndarray, dist: np.ndarray, camera_name: str
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


def save_calibration_yaml(
    filename: str,
    image_size: tuple[int, int],
    mtx: np.ndarray,
    dist: np.ndarray,
    camera_name: str,
):
    with open(filename, "w") as f:
        yaml_string = create_yaml_string(image_size, mtx, dist, camera_name)
        f.write(yaml_string)


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
    print(create_yaml_string((1920, 1080), mtx, dist, "test"))

    print(format_array(mtx))