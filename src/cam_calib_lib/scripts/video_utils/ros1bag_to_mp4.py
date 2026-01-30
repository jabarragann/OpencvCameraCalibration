#!/usr/bin/env python3

import rospy
import subprocess
import os
import signal

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

import message_filters


class StereoBagExtractor:
    def __init__(
        self,
        bag_path,
        left_topic,
        right_topic,
        output_dir,
        queue_size=10,
        slop=0.02,
    ):
        self.bag_path = bag_path
        self.bridge = CvBridge()
        self.count = 0

        self.left_dir = os.path.join(output_dir, "left")
        self.right_dir = os.path.join(output_dir, "right")
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

        rospy.loginfo("Initialized stereo extractor")

    def callback(self, left_msg, right_msg):
        try:
            left_cv = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding="bgr8")
            right_cv = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr(f"cv_bridge error: {e}")
            return

        left_path = os.path.join(self.left_dir, f"left_{self.count:06d}.jpg")
        right_path = os.path.join(self.right_dir, f"right_{self.count:06d}.jpg")

        cv2.imwrite(left_path, left_cv)
        cv2.imwrite(right_path, right_cv)

        rospy.loginfo(f"Saved pair {self.count}")
        self.count += 1


def play_rosbag(bag_path):
    return subprocess.Popen(
        ["rosbag", "play", "--clock", bag_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )


def main():
    rospy.init_node("stereo_bag_extractor", anonymous=True)

    bag_path = rospy.get_param("~bag_path")
    left_topic = rospy.get_param("~left_topic", "/left/image_raw")
    right_topic = rospy.get_param("~right_topic", "/right/image_raw")
    output_dir = rospy.get_param("~output_dir", "./stereo_images")

    extractor = StereoBagExtractor(
        bag_path=bag_path,
        left_topic=left_topic,
        right_topic=right_topic,
        output_dir=output_dir,
    )

    bag_process = play_rosbag(bag_path)

    rospy.loginfo("Playing rosbag...")
    rospy.spin()

    # Cleanup on shutdown
    rospy.loginfo("Stopping rosbag")
    os.killpg(os.getpgid(bag_process.pid), signal.SIGTERM)


if __name__ == "__main__":
    main()