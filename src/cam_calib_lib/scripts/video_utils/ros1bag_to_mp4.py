#!/usr/bin/env python3

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import rospy
import subprocess
import os
import signal

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

import message_filters
from tap import Tap


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


class SimpleArgumentParser(Tap):
    bag_path: Path = Path("output/mockor_calibration/calibration_data03.bag")
    left_topic: str = "/decklink_left/camera/image_raw"
    right_topic: str = "/decklink_right/camera/image_raw"
    output_dir: Path = Path("output/microscope_calibration/images")


class StereoBagExtractor2:
    def __init__(
        self,
        bag_path,
        left_topic,
        right_topic,
        output_dir,
        queue_size=10,
        slop=0.05,
        args: Optional[SimpleArgumentParser] = None,
    ):
        self.bag_path = bag_path
        self.bridge = CvBridge()
        self.count = 0

        self.left_dir = output_dir / "left"
        self.right_dir = output_dir / "right"
        os.makedirs(self.left_dir, exist_ok=True)
        os.makedirs(self.right_dir, exist_ok=True)

        # Subscribers with message_filters
        left_sub = message_filters.Subscriber(left_topic, Image)
        right_sub = message_filters.Subscriber(right_topic, Image)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [left_sub, right_sub],
            queue_size=queue_size,
            slop=slop,
            allow_headerless=False,
        )
        self.sync.registerCallback(self.callback)

        ## ffmpeg processes
        assert args is not None, "args must be provided for ffmpeg initialization"
        width = 1920
        height = 1080
        fps= 30
        self.left_writer_proc = start_ffmpeg(
            str(args.output_dir / "left.mp4"), width, height, fps)

        self.right_writer_proc = start_ffmpeg(
            str(args.output_dir / "right.mp4"), width, height, fps
        )
        rospy.loginfo("Initialized stereo extractor")

    def callback(self, left_msg, right_msg):
        try:
            left_cv = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding="bgr8")
            right_cv = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr(f"cv_bridge error: {e}")
            return

        self.left_writer_proc.stdin.write(left_cv.tobytes())  # type: ignore
        self.right_writer_proc.stdin.write(right_cv.tobytes())  # type: ignore
        # left_path = os.path.join(self.left_dir, f"left_{self.count:06d}.jpg")
        # right_path = os.path.join(self.right_dir, f"right_{self.count:06d}.jpg")

        # cv2.imwrite(str(left_path), left_cv)
        # cv2.imwrite(str(right_path), right_cv)

        rospy.loginfo(f"Saved pair {self.count}")
        self.count += 1


def play_rosbag(bag_path):
    return subprocess.Popen(
        ["rosbag", "play", "--clock", str(bag_path)],
        # stdout=subprocess.DEVNULL,
        # stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )


def main():
    args = SimpleArgumentParser().parse_args()
    rospy.init_node("stereo_bag_extractor", anonymous=True)
    print("rosbag to video script")

    # args = CliArgs().parse_args()
    # bag_path = rospy.get_param("~bag_path", "output/microscope_calibration/calib4-01302025/calibration_data03.bag")
    # left_topic = rospy.get_param("~left_topic", "/left/image_raw")
    # right_topic = rospy.get_param("~right_topic", "/right/image_raw")
    # output_dir = rospy.get_param("~output_dir", "output/microscope_calibration/calib4-01302025/images")

    extractor = StereoBagExtractor2(
        bag_path=args.bag_path,
        left_topic=args.left_topic,
        right_topic=args.right_topic,
        output_dir=args.output_dir,
        args=args,
    )

    bag_process = play_rosbag(args.bag_path)

    rospy.loginfo("Playing rosbag...")
    rospy.spin()

    # Cleanup on shutdown
    rospy.loginfo("Stopping rosbag")
    os.killpg(os.getpgid(bag_process.pid), signal.SIGTERM)


    # Cleanup
    extractor.left_writer_proc.stdin.close()  # type: ignore
    extractor.left_writer_proc.wait()
    extractor.right_writer_proc.stdin.close()  # type: ignore
    extractor.right_writer_proc.wait()

if __name__ == "__main__":
    main()
