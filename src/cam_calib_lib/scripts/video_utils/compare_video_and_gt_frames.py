"""
A video is generated from a rosbag and then frames are extracted.
Compare against ground truth frames to ensure no desyncronization with video encoding.

"""

from tap import Tap
from pathlib import Path
import cv2
import numpy as np
import numpy.typing as npt
from natsort import natsorted


class CliArgs(Tap):
    gt_frames: Path
    video_frames: Path


def verify_frame(
    gt_frame: npt.NDArray[np.uint8],
    extracted_frame: npt.NDArray[np.uint8],
    tol: int = 0,
) -> tuple[bool, float]:
    """
    Compare a ground-truth frame to an extracted frame.
    """

    # Check shape first
    if gt_frame.shape != extracted_frame.shape:
        print(
            f"Shape mismatch: GT {gt_frame.shape} vs Extracted {extracted_frame.shape}"
        )
        return False, 0.0

    # Compute absolute difference
    diff = np.abs(gt_frame.astype(np.int32) - extracted_frame.astype(np.int32))

    # Check if all differences are <= tolerance
    if np.all(diff <= tol):
        return True
    else:
        # Optional: report some stats
        max_diff = diff.max()
        mean_diff = diff.mean()
        print(f"Frame mismatch: max_diff={max_diff}, mean_diff={mean_diff:.2f}")
        return False, mean_diff


def main(args: CliArgs):
    gt_frames = [frame_path for frame_path in args.gt_frames.glob("*.png")]
    gt_frames = natsorted(gt_frames)

    gt_frames_names = [frame_path.name for frame_path in gt_frames]

    extracted_frames = [frame_path for frame_path in args.video_frames.glob("*.png")]
    extracted_frames = natsorted(extracted_frames)

    extracted_frames_filtered = [
        frame_path
        for frame_path in extracted_frames
        if frame_path.name in gt_frames_names
    ]
    extracted_frames_names = [frame_path.name for frame_path in extracted_frames_filtered]

    gt_frames_filtered = [
        frame_path
        for frame_path in gt_frames
        if frame_path.name in extracted_frames_names
    ]

    print(
        f"Found {len(gt_frames_filtered)} gt frames and {len(extracted_frames_filtered)} extracted frames"
    )

    for frame_idx, (gt_frame_path, extracted_frame_path) in enumerate(
        zip(gt_frames_filtered, extracted_frames_filtered)
    ):
        print(f"Comparing frame {frame_idx}: {gt_frame_path.name} and {extracted_frame_path.name}")
        match, diff = verify_frame(
            cv2.imread(gt_frame_path), cv2.imread(extracted_frame_path)
        )
        print(f"Frame {frame_idx}: {match}, diff: {diff:.05f}")


if __name__ == "__main__":
    args = CliArgs().parse_args()
    main(args)
