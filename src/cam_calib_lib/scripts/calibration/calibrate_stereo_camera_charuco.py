import csv
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from cam_calib_lib import ArgumentParsers as calib_args
from cam_calib_lib import Utils as calib_utils
from cam_calib_lib.ArgumentParsers import CameraCalibrationCharucoArgs
from cam_calib_lib.OpencvBoards import OpenCVCharucoBoard
from cam_calib_lib.scripts.calibration.calibrate_stereo_camera import (
    display_undistorted_images,
    load_stereo_images,
    parse_frame_ids,
    rectify_and_display,
)
from cam_calib_lib.StereoCalibration import stereo_calibration_pipeline
from cam_calib_lib.Utils import index_to_bgr


def frame_id(image_path: Path) -> str:
    """Original frame number from a left-XXXX.png / right-XXXX.png filename, e.g. '0003'."""
    return image_path.stem.split("-", 1)[-1].split("_", 1)[-1]


class CliArgs(CameraCalibrationCharucoArgs):
    print_debug: bool = False
    save_debug_images: bool = False
    exclude: list[str] = []  # frame ids to skip, e.g. --exclude "0034 0048"
    show_windows: bool = False  # open OpenCV windows with undistorted/rectified previews (needs a display)
    rect_line_spacing: int = 50  # min vertical spacing (px) between epipolar lines in rect_vis
    min_corners: int = 20  # min corners seen in both images for a pair to be used in calibration


def prepare_stereo_charuco_data(
    board: OpenCVCharucoBoard,
    left_images: list[Path],
    right_images: list[Path],
    min_corners: int = 20,
) -> tuple[
    list[npt.NDArray[np.float32]],  # obj_points
    list[npt.NDArray[np.float32]],  # left_img_points
    list[npt.NDArray[np.float32]],  # right_img_points
    list[npt.NDArray[np.uint8]],  # left_valid_images
    list[npt.NDArray[np.uint8]],  # right_valid_images
    list[str],  # valid_frame_ids
]:
    """
    Prepare object points and image points for stereo calibration.

    Returns:
        (obj_points, left_img_points, right_img_points, valid_pairs)
    """

    obj_points: list[npt.NDArray[np.float32]] = []
    left_img_points: list[npt.NDArray[np.float32]] = []
    right_img_points: list[npt.NDArray[np.float32]] = []
    left_valid_images: list[npt.NDArray[np.uint8]] = []
    right_valid_images: list[npt.NDArray[np.uint8]] = []
    valid_frame_ids: list[str] = []

    print("Detecting chessboard corners...")
    for idx, (left_path, right_path) in enumerate(zip(left_images, right_images)):
        left_img = cv2.imread(str(left_path))
        right_img = cv2.imread(str(right_path))
        fid = frame_id(left_path)

        if left_img is None or right_img is None:
            print(f"Warning: Could not load image pair {idx} (frame {fid})")
            continue

        left_img = np.asarray(left_img, dtype=np.uint8)
        right_img = np.asarray(right_img, dtype=np.uint8)

        # Detect corners in both images
        left_ret, left_corners, left_corners_ids = board.detect_corners(left_img)
        right_ret, right_corners, right_corners_ids = board.detect_corners(right_img)

        valid_corners_in_image = 0
        if left_ret and right_ret:
            # Get indices in left_corners_ids that are also present in right_corners_ids
            left_common_idx = np.isin(
                left_corners_ids.flatten(), right_corners_ids.flatten()
            ).nonzero()[0]
            left_corners_filtered = left_corners[left_common_idx]
            left_corners_ids_filtered = left_corners_ids[left_common_idx]

            right_common_idx = np.isin(
                right_corners_ids.flatten(), left_corners_ids.flatten()
            ).nonzero()[0]
            right_corners_filtered = right_corners[right_common_idx]
            right_corners_ids_filtered = right_corners_ids[right_common_idx]
            valid_corners_in_image = right_corners_filtered.shape[0]

            if valid_corners_in_image >= min_corners:
                objp_left = board.get_object_points(
                    left_corners_filtered, left_corners_ids_filtered
                )
                objp_right = board.get_object_points(
                    right_corners_filtered, right_corners_ids_filtered
                )

                assert (
                    objp_left.shape == objp_right.shape
                ), "Object point arrays have different shapes"
                assert np.allclose(
                    objp_left, objp_right
                ), "Object points for left and right images differ"

                obj_points.append(objp_left)
                left_img_points.append(left_corners_filtered)
                right_img_points.append(right_corners_filtered)
                left_valid_images.append(left_img)
                right_valid_images.append(right_img)
                valid_frame_ids.append(fid)

        if valid_corners_in_image >= min_corners:
            print(
                f"  Pair {idx} (frame {fid}): ✓ Both images detected. Added {valid_corners_in_image} object points"
            )
        else:
            print(
                f"  Pair {idx} (frame {fid}): ✗ {valid_corners_in_image} corners detected. "
                + f"Minimum {min_corners} corners required"
            )

    print(
        f"\nSuccessfully detected corners in {len(obj_points)}/{len(left_images)} pairs"
    )
    return (
        obj_points,
        left_img_points,
        right_img_points,
        left_valid_images,
        right_valid_images,
        valid_frame_ids,
    )


def save_detected_corners(
    output_path: Path,
    obj_points: list[npt.NDArray[np.float32]],
    left_img_points: list[npt.NDArray[np.float32]],
    right_img_points: list[npt.NDArray[np.float32]],
    left_valid_images: list[npt.NDArray[np.uint8]],
    right_valid_images: list[npt.NDArray[np.uint8]],
    frame_ids: list[str],
    radius: int = 4,
    thickness: int = -1,
):
    output_path.mkdir(parents=True, exist_ok=True)

    num_views = len(left_img_points)

    for i in range(num_views):
        imgL = left_valid_images[i].copy()
        imgR = right_valid_images[i].copy()

        if imgL.ndim == 2:
            imgL = cv2.cvtColor(imgL, cv2.COLOR_GRAY2BGR)
        if imgR.ndim == 2:
            imgR = cv2.cvtColor(imgR, cv2.COLOR_GRAY2BGR)

        ptsL = left_img_points[i].reshape(-1, 2)
        ptsR = right_img_points[i].reshape(-1, 2)

        assert ptsL.shape == ptsR.shape, "Left/right corner count mismatch"

        n_pts = ptsL.shape[0]

        for j in range(n_pts):
            color = index_to_bgr(j, n_pts)

            pL = tuple(np.round(ptsL[j]).astype(int))
            pR = tuple(np.round(ptsR[j]).astype(int))

            cv2.circle(imgL, pL, radius, color, thickness)
            cv2.circle(imgR, pR, radius, color, thickness)

            # if draw_indices:
            #     cv2.putText(
            #         imgL, str(j), (pL[0] + 5, pL[1] - 5),
            #         cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            #     )
            #     cv2.putText(
            #         imgR, str(j), (pR[0] + 5, pR[1] - 5),
            #         cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            #     )

        # Concatenate for easier visual inspection
        debug = np.hstack([imgL, imgR])

        out_file = output_path / f"corners_{frame_ids[i]}.png"
        cv2.imwrite(str(out_file), debug)


def save_rectified_corner_lines(
    board: OpenCVCharucoBoard,
    left_img: npt.NDArray[np.uint8],
    right_img: npt.NDArray[np.uint8],
    left_map1: npt.NDArray[np.float32],
    left_map2: npt.NDArray[np.float32],
    right_map1: npt.NDArray[np.float32],
    right_map2: npt.NDArray[np.float32],
    save_vis_path: Path,
    min_line_spacing: int = 50,
    radius: int = 6,
) -> list[float]:
    """
    Rectify a stereo pair, detect ChArUco corners in both rectified images and, for
    corners seen in both, draw a horizontal line from the left corner across both images.
    If rectification is correct, the matching right corner lands on the same line.

    Lines are kept at least `min_line_spacing` px apart to avoid overcrowding; only corners
    that got a line are drawn. Returns |dy| (px) for every matched corner (empty if none).
    """
    left_rect = cv2.remap(left_img, left_map1, left_map2, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_img, right_map1, right_map2, cv2.INTER_LINEAR)
    left_rect = np.asarray(left_rect, dtype=np.uint8)
    right_rect = np.asarray(right_rect, dtype=np.uint8)

    left_ret, left_corners, left_ids = board.detect_corners(left_rect)
    right_ret, right_corners, right_ids = board.detect_corners(right_rect)

    vis = np.hstack([left_rect, right_rect])
    w = left_rect.shape[1]

    if not (left_ret and right_ret):
        cv2.putText(vis, "No common corners", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (139, 0, 0), 3)
        cv2.imwrite(str(save_vis_path), vis)
        return []

    left_pts = dict(zip(left_ids.flatten(), left_corners.reshape(-1, 2)))
    right_pts = dict(zip(right_ids.flatten(), right_corners.reshape(-1, 2)))
    common_ids = [i for i in left_pts if i in right_pts]
    if not common_ids:
        cv2.imwrite(str(save_vis_path), vis)
        return []

    dys = [float(abs(left_pts[i][1] - right_pts[i][1])) for i in common_ids]

    # Greedy selection top-to-bottom: keep a corner only if its line is far enough
    # from the previously kept one.
    common_ids.sort(key=lambda i: left_pts[i][1])
    selected: list[int] = []
    for i in common_ids:
        if not selected or left_pts[i][1] - left_pts[selected[-1]][1] >= min_line_spacing:
            selected.append(i)

    for k, i in enumerate(selected):
        color = index_to_bgr(k, len(selected))
        pl = left_pts[i]
        pr = right_pts[i]
        y = int(round(pl[1]))
        cv2.line(vis, (0, y), (2 * w, y), color, 1, cv2.LINE_AA)
        cv2.circle(vis, (int(round(pl[0])), y), radius, color, 2, cv2.LINE_AA)
        cv2.circle(vis, (int(round(pr[0])) + w, int(round(pr[1]))), radius, color, 2, cv2.LINE_AA)

    mean_dy = float(np.mean(dys))
    cv2.putText(
        vis,
        f"matched corners: {len(common_ids)}  mean |dy|: {mean_dy:.2f}px  std: {np.std(dys):.2f}px  max |dy|: {max(dys):.2f}px",
        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (139, 0, 0), 2, cv2.LINE_AA,
    )
    cv2.imwrite(str(save_vis_path), vis)
    return dys


def main(args: CliArgs):
    assert isinstance(args.images_path, Path)
    assert isinstance(args.output_path, Path)
    assert args.images_path.exists(), f"Images path does not exist: {args.images_path}"

    # args.save(args.output_path / "input_parameters.yaml")

    calib_utils.save_calibration_parameters_charuco(
        args.output_path / "input_parameters.yaml",
        args.aruco_dict,
        args.squares_in_x_dim,
        args.squares_in_y_dim,
        args.square_size,
        args.marker_size,
        args.distortion_flags,
        args.stereo_flags,
        excluded_frames=sorted(parse_frame_ids(args.exclude)),
        rectify_flags=args.rectify_flags,
    )

    ## Opencv Flags
    distortion_flags = calib_args.compose_cv2_flags(args.distortion_flags)
    stereo_flags = calib_args.compose_cv2_flags(args.stereo_flags)
    rectify_flags = calib_args.compose_cv2_flags(args.rectify_flags)
    aruco_dict_flag = calib_args.aruco_dict_from_string_string_list(args.aruco_dict)

    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_flag)
    board = OpenCVCharucoBoard(
        squares_in_x_dim=args.squares_in_x_dim,
        squares_in_y_dim=args.squares_in_y_dim,
        square_length=args.square_size,
        marker_length=args.marker_size,
        aruco_dict=aruco_dict,
    )

    # Load stereo image pairs
    left_images, right_images = load_stereo_images(args.images_path, exclude=args.exclude)

    if len(left_images) == 0:
        raise ValueError("No valid stereo image pairs found!")

    # Get image size from first image
    first_img = cv2.imread(str(left_images[0]))
    if first_img is None:
        raise ValueError(f"Could not load first image: {left_images[0]}")
    image_size = (first_img.shape[1], first_img.shape[0])

    # Prepare calibration data
    (
        obj_points,
        left_img_points,
        right_img_points,
        left_valid_images,
        right_valid_images,
        valid_frame_ids,
    ) = prepare_stereo_charuco_data(
        board, left_images, right_images, args.min_corners
    )

    if len(obj_points) < 10:
        print(
            f"Warning: Only {len(obj_points)} valid pairs found. At least 10 pairs are recommended for good calibration."
        )

    stereo_calibration_output, rectification_output, rectification_maps_output = (
        stereo_calibration_pipeline(
            obj_points,
            left_img_points,
            right_img_points,
            image_size,
            distortion_flags,
            stereo_flags,
            args.output_path,
            rectify_flags=rectify_flags,
        )
    )

    if args.show_windows:
        # Test rectification on first pair
        print("\nTesting rectification on first image pair...")
        left_img = cv2.imread(str(left_images[0]))
        right_img = cv2.imread(str(right_images[0]))

        if left_img is not None and right_img is not None:
            left_img = np.asarray(left_img, dtype=np.uint8)
            right_img = np.asarray(right_img, dtype=np.uint8)

            display_undistorted_images(
                left_img,
                right_img,
                stereo_calibration_output.left_mtx,
                stereo_calibration_output.left_dist,
                stereo_calibration_output.right_mtx,
                stereo_calibration_output.right_dist,
            )

            rectify_and_display(
                left_img,
                right_img,
                rectification_maps_output.left_map1,
                rectification_maps_output.left_map2,
                rectification_maps_output.right_map1,
                rectification_maps_output.right_map2,
            )

    print("\nCalibration complete!")

    if args.save_debug_images:
        print("Saving detected corners...")
        save_detected_corners(
            args.output_path / "debug_images" / "detected_corners",
            obj_points,
            left_img_points,
            right_img_points,
            left_valid_images,
            right_valid_images,
            valid_frame_ids,
        )

        # Only evaluate rectification on the pairs that were used in the calibration
        used_frame_ids = set(valid_frame_ids)
        used_pairs = [
            i for i in range(len(left_images)) if frame_id(left_images[i]) in used_frame_ids
        ]
        print(f"Saving rectified images for {len(used_pairs)} pairs used in calibration...")
        save_vis_path = args.output_path / "debug_images" / "rect_vis"
        save_vis_path.mkdir(parents=True, exist_ok=True)
        report_rows: list[dict[str, object]] = []
        all_dys: list[float] = []
        per_image_dys: list[list[float]] = []
        for i in used_pairs:
            left_img = np.asarray(cv2.imread(str(left_images[i])), dtype=np.uint8)
            right_img = np.asarray(cv2.imread(str(right_images[i])), dtype=np.uint8)
            dys = save_rectified_corner_lines(
                board,
                left_img,
                right_img,
                rectification_maps_output.left_map1,
                rectification_maps_output.left_map2,
                rectification_maps_output.right_map1,
                rectification_maps_output.right_map2,
                save_vis_path=save_vis_path / f"rectified_stereo_pair_{frame_id(left_images[i])}.png",
                min_line_spacing=args.rect_line_spacing,
            )
            all_dys.extend(dys)
            per_image_dys.append(dys)
            report_rows.append(
                {
                    "pair": i,
                    "frame": frame_id(left_images[i]),
                    "left_image": left_images[i].name,
                    "right_image": right_images[i].name,
                    "num_corners": len(dys),
                    "mean_dy_px": f"{np.mean(dys):.4f}" if dys else "",
                    "std_dy_px": f"{np.std(dys):.4f}" if dys else "",
                    "median_dy_px": f"{np.median(dys):.4f}" if dys else "",
                    "max_dy_px": f"{np.max(dys):.4f}" if dys else "",
                }
            )

        report_path = args.output_path / "rectification_errors.csv"
        with open(report_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
            writer.writeheader()
            writer.writerows(report_rows)
        print(f"Saved per-image rectification errors to {report_path}")

        if all_dys:
            # Macro: one error per image (its mean |dy|), then stats over images.
            # Micro: stats over all corners pooled, so images with more corners weigh more.
            per_image = [float(np.mean(d)) for d in per_image_dys if d]
            print(f"Rectification |dy| ({len(per_image)} pairs, {len(all_dys)} corners):")
            for name, vals in [("macro (per image)", per_image), ("micro (per corner)", all_dys)]:
                print(
                    f"  {name:<19} min {np.min(vals):.2f}px, max {np.max(vals):.2f}px, "
                    f"mean {np.mean(vals):.2f}px, std {np.std(vals):.2f}px, median {np.median(vals):.2f}px"
                )
        else:
            print("Rectification |dy|: no corners matched in any rectified pair")


if __name__ == "__main__":
    args = CliArgs().parse_args()
    main(args)
