from pathlib import Path
import cv2
from tap import Tap

from cam_calib_lib.StereoCalibration import compute_rectification_maps

from cam_calib_lib.Utils import load_camera_calib
from cam_calib_lib.scripts.calibration.calibrate_stereo_camera import load_stereo_images, rectify_and_display


class CliArgs(Tap):
    images_path: Path
    output_path: Path
    left_calibration_file: Path
    right_calibration_file: Path





def main(args: CliArgs):
    # K, D = load_camera_calibration(args.images_path/ f"{args.camera}.yaml")
    params_dict_left = load_camera_calib(args.left_calibration_file)
    parms_dict_right = load_camera_calib(args.right_calibration_file)

    mtx_left = params_dict_left["camera_matrix"]
    dist_left = params_dict_left["dist_coeffs"]
    rec_matrix_left = params_dict_left["rectification_matrix"]
    projection_matrix_left = params_dict_left["projection_matrix"]

    mtx_right = parms_dict_right["camera_matrix"]
    dist_right = parms_dict_right["dist_coeffs"]
    rec_matrix_right = parms_dict_right["rectification_matrix"]
    projection_matrix_right = parms_dict_right["projection_matrix"]
    image_size = params_dict_left["image_size"]

    left_map1, left_map2, right_map1, right_map2 = compute_rectification_maps(
        rec_matrix_left,
        rec_matrix_right,
        projection_matrix_left,
        projection_matrix_right,
        image_size,
        mtx_left,
        dist_left,
        mtx_right,
        dist_right,
    )

    left_images, right_images = load_stereo_images(args.images_path)
    save_vis_path = args.output_path / "rectified_stereo_pairs"
    save_vis_path.mkdir(parents=True, exist_ok=True)

    for i in range(len(left_images)):
        left_img = cv2.imread(str(left_images[i]))
        right_img = cv2.imread(str(right_images[i]))

        rectify_and_display(
            left_img,
            right_img,
            left_map1,
            left_map2,
            right_map1,
            right_map2,
            display=False,
            save_vis_path=save_vis_path / f"rectified_stereo_pair_{i}.png",
        )

    # regular_chessboard = OpenCVChessboard(
    #     corners_per_row=args.corners_per_board_row,
    #     corners_per_column=args.corners_per_board_column,
    #     square_size=args.square_size,
    # )

    # obj_points, img_points = prepare_calibration_data(regular_chessboard, images)

    # rvecs, tvecs = calculate_rvecs_tvecs(obj_points, img_points, mtx, dist)

    # # # print(f"tvecs[0]: {tvecs[0]}")
    # # # print(f"rvecs[0]: {rvecs[0]}")

    # # # projected_points, _ = cv2.projectPoints(
    # # #     obj_points[0], rvecs[0], tvecs[0], mtx, dist
    # # # )
    # # projected_points = projected_points.reshape(-1, 2)
    # # print(f"obj_points[0]: {obj_points[0][0:5]}")
    # # print(f"projected_points: {projected_points[0:5]}")

    # # error_with_distortion = calculate_reprojection_error(
    # #     obj_points, img_points, rvecs, tvecs, mtx, dist
    # # )
    # # error_without_distortion = calculate_reprojection_error(
    # #     obj_points,
    # #     img_points,
    # #     rvecs,
    # #     tvecs,
    # #     mtx,
    # #     np.zeros_like(dist),
    # # )

    # print(f"Reprojection error (with distortion): {error_with_distortion:.4f} pixels")
    # print(
    #     f"Reprojection error (without distortion): {error_without_distortion:.4f} pixels"
    # )

    # print(
    #     f"\nDisplaying projected vs ground truth corners for first {args.camera_name} image..."
    # )
    # first_img = cv2.imread(str(images[0]))

    # if first_img is not None and len(obj_points) > 0:
    #     first_img = np.asarray(first_img, dtype=np.uint8)
    #     # Use the first valid image's data
    #     display_projected_corners(
    #         first_img,
    #         obj_points[0],
    #         img_points[0],
    #         rvecs[0],
    #         tvecs[0],
    #         mtx,
    #         dist,
    #         args.camera_name,
    #         use_distortion=True,
    #     )


if __name__ == "__main__":
    args = CliArgs().parse_args()
    main(args)
