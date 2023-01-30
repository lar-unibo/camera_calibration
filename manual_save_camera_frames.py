#!/usr/bin/env python

import os
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
import tf2_ros

from cv_bridge import CvBridge


class SaveFrames():

    def __init__(self):

        self.camera_topic = 'camera/color/image_raw'
        self.image_folder = "/tmp/manual_saved_frames"
        self.ee_frame = "panda_hand"

        os.makedirs(self.image_folder)

        rospy.wait_for_message(self.camera_topic, Image)
        print('Camera Topic : OK')

        self.bridge = CvBridge()
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

    def save_img_and_pose(self, counter):
        
        rospy.sleep(0.5)
        data = rospy.wait_for_message(self.camera_topic, Image)
        img = cv2.cvtColor(self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough'), cv2.COLOR_BGR2RGB)

        try:
            T = self.tfBuffer.lookup_transform("world", self.ee_frame, rospy.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            pass

        pose = [T.transform.translation.x, T.transform.translation.y, T.transform.translation.z,
                T.transform.rotation.x, T.transform.rotation.y, T.transform.rotation.z, T.transform.rotation.w]
        
        cv2.imwrite(os.path.join(self.image_folder, "frame_ " + str(counter) + ".png"), img)
        np.savetxt(os.path.join(self.image_folder, "frame_ " + str(counter) + ".txt"), np.array(pose).reshape(1,-1))





if __name__ == '__main__' :

    rospy.init_node('save_camera_frames')

    #######################################
    # Init
    ####################################### 
    f = SaveFrames()    

    counter = 0
    while not rospy.is_shutdown():
        input("Press ENTER to save frame {}".format(counter))
        f.save_img_and_pose(counter)
        counter += 1
