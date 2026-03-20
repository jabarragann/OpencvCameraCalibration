"""
Convert a ROS 2 bag into frame-exact stereo MP4 videos and a CSV of timestamps.
Each CSV row maps 1:1 to a frame in both videos; frames are streamed to FFmpeg
(H.264, constant frame rate) to prevent frame drops or resampling.

Verification commands
---------------------
Verify that the number of frames in each video matches the CSV:

# Number of frames in left video
ffprobe -v error -count_frames \
    -select_streams v:0 \
    -show_entries stream=nb_read_frames \
    -of default=nokey=1:noprint_wrappers=1 output/left.mp4  <-- Change path

# Number of frames in CSV (subtract header)
wc -l output/timestamps.csv  <-- Change path

"""

import csv
import os
import subprocess
from pathlib import Path

from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message
from tap import Tap


class CliArgs(Tap):
    bag_path: Path
    output_path: Path = Path("./output_videos")
    left_topic: str = "/dvrk_video/left/image_raw"
    right_topic: str = "/dvrk_video/right/image_raw"
    fps: int = 30


def start_ffmpeg(output_path, width, height, fps):
    # fmt: off
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "pipe:0",
        "-an", "-c:v", "libx264",
        "-preset", "slow", "-crf", "18", "-pix_fmt",
        "yuv420p", "-vsync", "cfr", "-x264-params", "keyint=1",
        output_path,
    ]
    # fmt: on
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def main(args: CliArgs):
    assert isinstance(args.bag_path, Path)
    assert isinstance(args.output_path, Path)
    assert args.bag_path.exists(), f"Bag path does not exist: {args.bag_path}"

    args.output_path.mkdir(parents=True, exist_ok=True)

    bridge = CvBridge()

    # --- Open rosbag ---
    reader = SequentialReader()
    storage_options = StorageOptions(
        uri=str(args.bag_path),
        storage_id="mcap",
    )
    converter_options = ConverterOptions("", "")
    reader.open(storage_options, converter_options)

    topic_types = {
        topic.name: topic.type for topic in reader.get_all_topics_and_types()
    }

    timestamps = []
    frame_idx = 0

    # Temporary storage for pairing
    left_frame = None
    right_frame = None
    left_ts = None
    right_ts = None

    ## ffmpeg processes
    left_writer_proc = None
    right_writer_proc = None

    initialize_ffmpeg_process = False

    while reader.has_next():
        topic, data, _ = reader.read_next()
        msg_type = get_message(topic_types[topic])
        msg = deserialize_message(data, msg_type)

        if topic == args.left_topic:
            left_frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            left_ts = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

        elif topic == args.right_topic:
            right_frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            right_ts = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

        # Once both frames are available → commit frame
        if left_frame is not None and right_frame is not None:

            if not initialize_ffmpeg_process:
                width = left_frame.shape[1]
                height = left_frame.shape[0]

                left_writer_proc = start_ffmpeg(
                    str(args.output_path / "left.mp4"), width, height, args.fps
                )
                right_writer_proc = start_ffmpeg(
                    str(args.output_path / "right.mp4"), width, height, args.fps
                )

                initialize_ffmpeg_process = True

            left_writer_proc.stdin.write(left_frame.tobytes())  # type: ignore
            right_writer_proc.stdin.write(right_frame.tobytes())  # type: ignore
            timestamps.append([frame_idx, left_ts, right_ts])

            frame_idx += 1
            left_frame = right_frame = None
            left_ts = right_ts = None

    # Cleanup
    if left_writer_proc:
        left_writer_proc.stdin.close()  # type: ignore
        left_writer_proc.wait()
    if right_writer_proc:
        right_writer_proc.stdin.close()  # type: ignore
        right_writer_proc.wait()

    # --- Write timestamps ---
    with open(os.path.join(args.output_path, "timestamps.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "left_timestamp_ns", "right_timestamp_ns"])
        writer.writerows(timestamps)

    print(f"Done. Extracted {frame_idx} stereo frames.")


if __name__ == "__main__":
    args = CliArgs().parse_args()
    main(args)
