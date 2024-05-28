#!/usr/bin/env python3

import os
import cv2
import numpy as np
import rclpy
import rclpy.time
from sensor_msgs.msg import Image
import tf2_ros
from rclpy.node import Node
from cv_bridge import CvBridge
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup


class SaveFrames(Node):

    def __init__(self):
        super().__init__('my_node')
        self.camera_topic = '/rs/rs/color/image_raw'
        self.image_folder = "/home/lar/ros2/official_ws/src/camera_images"
        self.ee_frame = "ur_right_flange"

        os.makedirs(self.image_folder)
        self.callback_group_1   = ReentrantCallbackGroup()
        self.timer = self.create_timer(0.001, self.current_pose, callback_group=self.callback_group_1)  

        self.sub = self.create_subscription(Image, self.camera_topic, self.take_image, 1)
        print('Camera Topic : OK')

        self.counter = 0
        self.bridge = CvBridge()
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=True) 

    def take_image(self, msg):
        self.data = msg

    def current_pose(self):
        origin_frame = "world"
        dest_frame = self.ee_frame

        # print("Getting pose")
        
        try:
            # Get the transform between /map and /base_footprint
            T = self._tf_buffer.lookup_transform(origin_frame, dest_frame,rclpy.time.Time())

            self.pose = [T.transform.translation.x, T.transform.translation.y, T.transform.translation.z,
                T.transform.rotation.x, T.transform.rotation.y, T.transform.rotation.z, T.transform.rotation.w]
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'Transform lookup failed: {e}')

    def save_img_and_pose(self):

        while rclpy.ok():
            input("Press ENTER to save frame {}".format(self.counter))

            img = cv2.cvtColor(self.bridge.imgmsg_to_cv2(self.data, desired_encoding='passthrough'), cv2.COLOR_BGR2RGB)
            
            cv2.imwrite(os.path.join(self.image_folder, "frame_ " + str(self.counter) + ".png"), img)
            np.savetxt(os.path.join(self.image_folder, "frame_ " + str(self.counter) + ".txt"), np.array(self.pose).reshape(1,-1))
            self.counter += 1

def main(args=None):
    rclpy.init(args=args)
    f = SaveFrames() 

    executor = MultiThreadedExecutor(num_threads=4)
    f.save_img_and_pose()
    executor.add_node(f)
    executor.spin()
    
    #Destroy the node explicitly
    executor.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__' :
    main()
    
   