from cam_calib_lib.ArgumentParsers import (
    aruco_dict_from_string_string_list,
    cv2_flag_from_string,
    compose_cv2_flags,
)
import cv2


def test_aruco_dict_from_string_list():
    assert aruco_dict_from_string_string_list("DICT_4X4_250") == cv2.aruco.DICT_4X4_250
    assert aruco_dict_from_string_string_list("DICT_5X5_100") == cv2.aruco.DICT_5X5_100
    assert aruco_dict_from_string_string_list("DICT_6X6_250") == cv2.aruco.DICT_6X6_250
    assert aruco_dict_from_string_string_list("DICT_7X7_250") == cv2.aruco.DICT_7X7_250


def test_cv2_flag_from_string():
    assert cv2_flag_from_string("CALIB_FIX_K3") == cv2.CALIB_FIX_K3
    assert (
        cv2_flag_from_string("CALIB_ZERO_TANGENT_DIST") == cv2.CALIB_ZERO_TANGENT_DIST
    )
    assert cv2_flag_from_string("CALIB_FIX_INTRINSIC") == cv2.CALIB_FIX_INTRINSIC
    assert cv2_flag_from_string("CALIB_FIX_ASPECT_RATIO") == cv2.CALIB_FIX_ASPECT_RATIO
    assert cv2_flag_from_string("CALIB_FIX_FOCAL_LENGTH") == cv2.CALIB_FIX_FOCAL_LENGTH
    assert (
        cv2_flag_from_string("CALIB_FIX_PRINCIPAL_POINT")
        == cv2.CALIB_FIX_PRINCIPAL_POINT
    )
    assert cv2_flag_from_string("CALIB_FIX_TANGENT_DIST") == cv2.CALIB_FIX_TANGENT_DIST
    assert cv2_flag_from_string("CALIB_FIX_K4") == cv2.CALIB_FIX_K4
    assert cv2_flag_from_string("CALIB_FIX_K5") == cv2.CALIB_FIX_K5
    assert cv2_flag_from_string("CALIB_FIX_K6") == cv2.CALIB_FIX_K6


def test_compose_cv2_flags1():
    composed_flags = ["CALIB_FIX_K3", "CALIB_ZERO_TANGENT_DIST", "CALIB_FIX_INTRINSIC"]

    final_flag = 0
    final_flag |= cv2.CALIB_FIX_K3
    final_flag |= cv2.CALIB_ZERO_TANGENT_DIST
    final_flag |= cv2.CALIB_FIX_INTRINSIC

    assert final_flag == compose_cv2_flags(composed_flags)


def test_compose_cv2_flags2():
    composed_flags = [
        "CALIB_FIX_K3",
        "CALIB_ZERO_TANGENT_DIST",
        "CALIB_FIX_INTRINSIC",
        "CALIB_FIX_ASPECT_RATIO",
    ]

    final_flag = 0
    final_flag |= cv2.CALIB_FIX_K3
    final_flag |= cv2.CALIB_ZERO_TANGENT_DIST
    final_flag |= cv2.CALIB_FIX_INTRINSIC
    final_flag |= cv2.CALIB_FIX_ASPECT_RATIO

    assert final_flag == compose_cv2_flags(composed_flags)
