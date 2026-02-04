from pathlib import Path
import traceback
import cv2
import numpy as np
import sys

from tap import Tap

from cam_calib_lib.Utils import load_stereo_camera_calib

def pretty_matrix(mat, title):
    np.set_printoptions(precision=3, suppress=True, floatmode="fixed")
    print(f"{title}:\n{mat}\n")

class ImageRectificationGUI:
    def __init__(self, left_image_path, right_image_path, stereo_calib_path):
        """
        Initialize the rectification GUI with left and right images.

        Args:
            left_image_path: Path to the left image
            right_image_path: Path to the right image
            stereo_calib_path: Path to the stereo calibration file
        """
        calibration_data = load_stereo_camera_calib(stereo_calib_path)
        self.left_mtx = calibration_data["left_camera_matrix"]
        self.left_dist = calibration_data["left_distortion_coefficients"]
        self.right_mtx = calibration_data["right_camera_matrix"]
        self.right_dist = calibration_data["right_distortion_coefficients"]
        self.F = calibration_data["fundamental_matrix"]
        self.E = calibration_data["essential_matrix"]
        self.T = calibration_data["T"]
        self.R = calibration_data["R"]

        # Load images
        self.left_img = cv2.imread(left_image_path)
        self.right_img = cv2.imread(right_image_path)

        self.image_size = (self.left_img.shape[1], self.left_img.shape[0])

        if self.left_img is None or self.right_img is None:
            raise ValueError("Could not load one or both images!")

        # Ensure images are the same size
        if self.left_img.shape != self.right_img.shape:
            print(
                "Warning: Images have different sizes. Resizing right image to match left."
            )

        self.alpha = 0  # Initial alpha value (0-100, will be divided by 100)
        self.window_name = "Stereo Image Rectification"

        # Create window and trackbar
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Alpha x100", self.window_name, 0, 100, self.on_alpha_change)

        # Initial rectification
        self.P1 = None
        self.P2 = None
        self.update_rectification()

    def on_alpha_change(self, alpha_val):
        """
        Callback function for trackbar changes.

        Args:
            alpha_val: Alpha value from trackbar (0-100)
        """
        self.alpha = alpha_val
        self.update_rectification()

    def update_rectification(self):
        """
        Perform stereo rectification with current alpha value and display results.
        """
        alpha = self.alpha / 100.0

        # Compute rectification transforms
        R1, R2, P1, P2, _, validRoi1, validRoi2 = cv2.stereoRectify(
            self.left_mtx,
            self.left_dist,
            self.right_mtx,
            self.right_dist,
            self.image_size,
            self.R,
            self.T,
            alpha=alpha,
            flags=cv2.CALIB_ZERO_DISPARITY,
        )
        self.P1 = P1
        self.P2 = P2

        # Compute rectification maps
        map1_left, map2_left = cv2.initUndistortRectifyMap(
            self.left_mtx, self.left_dist, R1, P1, self.image_size, cv2.CV_16SC2
        )

        map1_right, map2_right = cv2.initUndistortRectifyMap(
            self.right_mtx, self.right_dist, R2, P2, self.image_size, cv2.CV_16SC2
        )

        # Apply rectification
        rectified_left = cv2.remap(
            self.left_img, map1_left, map2_left, cv2.INTER_LINEAR
        )
        rectified_right = cv2.remap(
            self.right_img, map1_right, map2_right, cv2.INTER_LINEAR
        )

        # Draw horizontal lines to visualize rectification alignment
        rectified_left_lines = rectified_left.copy()
        rectified_right_lines = rectified_right.copy()

        h = rectified_left.shape[0]
        for y in range(0, h, 30):
            cv2.line(
                rectified_left_lines,
                (0, y),
                (rectified_left.shape[1], y),
                (0, 255, 0),
                1,
            )
            cv2.line(
                rectified_right_lines,
                (0, y),
                (rectified_right.shape[1], y),
                (0, 255, 0),
                1,
            )

        # Combine images side by side
        combined_original = np.hstack([self.left_img, self.right_img])
        combined_rectified = np.hstack([rectified_left_lines, rectified_right_lines])

        # Stack vertically to show before and after
        display = np.vstack([combined_original, combined_rectified])

        # Add text labels
        cv2.putText(
            display,
            "Original Images",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            display,
            f"Rectified Images (Alpha = {alpha:.2f})",
            (10, combined_original.shape[0] + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # Display
        cv2.imshow(self.window_name, display)
    
    def print_projection_matrices(self):
        pretty_matrix(self.P1, "Left Projection Matrix")
        pretty_matrix(self.P2, "Right Projection Matrix")

    def run(self):
        """
        Run the GUI main loop.
        """
        print("Stereo Image Rectification GUI")
        print("=" * 50)
        print("Controls:")
        print("  - Use trackbar to adjust Alpha value (0.0 - 1.0)")
        print("  - Alpha = 0: Zoom and crop to remove black areas")
        print("  - Alpha = 1: Keep all pixels, but with black areas")
        print("  - Press 'q' or ESC to quit")
        print("  - Press 's' to save rectified images")
        print("=" * 50)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:  # 'q' or ESC
                break
            elif key == ord("p"):
                self.print_projection_matrices()
            # elif key == ord("s"):
            #     self.save_rectified_images()

        cv2.destroyAllWindows()

    # Not tested
    # def save_rectified_images(self):
    #     """
    #     Save the current rectified images to disk.
    #     """
    #     alpha = self.alpha / 100.0

    #     # Recompute rectification (same as update_rectification but without display)
    #     R1, R2, P1, P2, Q, validRoi1, validRoi2 = cv2.stereoRectify(
    #         self.camera_matrix,
    #         self.dist_coeffs,
    #         self.camera_matrix,
    #         self.dist_coeffs,
    #         self.image_size,
    #         self.R,
    #         self.T,
    #         alpha=alpha,
    #         flags=cv2.CALIB_ZERO_DISPARITY,
    #     )

    #     map1_left, map2_left = cv2.initUndistortRectifyMap(
    #         self.camera_matrix, self.dist_coeffs, R1, P1, self.image_size, cv2.CV_16SC2
    #     )

    #     map1_right, map2_right = cv2.initUndistortRectifyMap(
    #         self.camera_matrix, self.dist_coeffs, R2, P2, self.image_size, cv2.CV_16SC2
    #     )

    #     rectified_left = cv2.remap(
    #         self.left_img, map1_left, map2_left, cv2.INTER_LINEAR
    #     )
    #     rectified_right = cv2.remap(
    #         self.right_img, map1_right, map2_right, cv2.INTER_LINEAR
    #     )

    #     # Save images
    #     left_filename = f"rectified_left_alpha_{alpha:.2f}.png"
    #     right_filename = f"rectified_right_alpha_{alpha:.2f}.png"

    #     cv2.imwrite(left_filename, rectified_left)
    #     cv2.imwrite(right_filename, rectified_right)

    #     print(f"Saved rectified images:")
    #     print(f"  - {left_filename}")
    #     print(f"  - {right_filename}")


class CliArgs(Tap):
    left_image_path: Path
    right_image_path: Path
    calib_path: Path

    def configure(self):
        self.add_argument("-l","--left-image-path", type=Path, required=True)
        self.add_argument("-r","--right-image-path", type=Path, required=True)
        self.add_argument("-c","--calib-path", type=Path, required=True)

def main():
    cli_args = CliArgs().parse_args()

    assert cli_args.left_image_path.exists(), (
        f"Left image path does not exist: {cli_args.left_image_path}"
    )
    assert cli_args.right_image_path.exists(), (
        f"Right image path does not exist: {cli_args.right_image_path}"
    )
    assert cli_args.calib_path.exists(), (
        f"Calibration path does not exist: {cli_args.calib_path}"
    )

    try:
        gui = ImageRectificationGUI(
            cli_args.left_image_path, cli_args.right_image_path, cli_args.calib_path
        )
        gui.run()
    except Exception as e:
        print(f"Error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
