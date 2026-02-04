import cv2
from matplotlib import pyplot as plt
import numpy as np
from cam_calib_lib.Utils import load_camera_calib
from pyprojroot.here import here


def replace_black_with_blue(img: np.ndarray) -> np.ndarray:
    # Create a copy to avoid modifying the input
    img_out = img.copy()

    # Define mask where all 3 channels are 0 (black)
    black_mask = np.all(img_out == [0, 0, 0], axis=-1)

    # Set those pixels to blue [255, 0, 0] in BGR
    img_out[black_mask] = [255, 0, 0]

    return img_out


def show_images(img1, img2, img3):
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))

    axs[0].imshow(img1)
    # axs[0].set_title(title1)
    axs[0].axis("off")

    axs[1].imshow(img2)
    # axs[1].set_title(title2)
    axs[1].axis("off")

    axs[2].imshow(img3)
    # axs[1].set_title(title2)
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    root_path = here()  # Path to root path from pyprojroot.
    img_path = root_path / "output/frames_5fps"
    camera_calib_file = img_path / "camera_calibration.yaml"

    calib_data = load_camera_calib(camera_calib_file)

    K = calib_data["camera_matrix"]
    dist = calib_data["dist_coeffs"]
    W, H = calib_data["image_size"]

    ## Juan notes on opencv documentation
    # Free scaling parameter between (0,1)
    # alpha=0 (when all the pixels in the undistorted image are valid) and
    # This means that the rectified images are zoomed and shifted so that only valid pixels are visible (no black areas after rectification).

    # alpha=1 (when all the source image pixels are retained in the undistorted image).
    # 1 means that the rectified image is decimated and shifted so that all the pixels from the original images from the cameras are retained
    # in the rectified images (no source image pixels are lost)

    free_scaling_param = 1
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        K, dist, (W, H), free_scaling_param, (W, H)
    )

    img = cv2.imread(str(img_path / "frame_001755.png"))
    img = replace_black_with_blue(img)
    # undistort
    mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, newcameramtx, (W, H), 5)  # type: ignore
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # crop the image
    x, y, w, h = roi
    dst2 = dst.copy()
    dst2 = dst2[y : y + h, x : x + w]

    print(f"free_scaling_param: {free_scaling_param}")
    print(f"Camera Matrix:\n{K}")
    print(f"Distortion Coefficients:\n{dist}")
    print(f"New Camera Matrix:\n{newcameramtx}")
    print(f"ROI: {roi}")

    show_images(img, dst, dst2)


if __name__ == "__main__":
    main()
