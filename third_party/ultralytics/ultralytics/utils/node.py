import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from collections import deque
import threading
import message_filters
import numpy as np    

color_queue = deque(maxlen=5)
depth_queue = deque(maxlen=5)
lock = threading.Lock()

class ListenerNode:  
    def __init__(self):  
        rospy.init_node('tracking', anonymous=True)
        self.bridge = CvBridge() 

        # Subscribe to the color image topic  
        # rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        # 订阅RGB和深度话题  
        rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)  
        depth_sub = message_filters.Subscriber('/camera/depth/image_raw', Image)  

        # 同步订阅器，确保RGB和深度图像同步  
        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 0.1)  
        self.ts.registerCallback(self.image_callback)  

    def image_callback(self, rgb_msg, depth_msg):  
        # Convert ROS Image message to OpenCV image
        rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")  
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")  
        global color_queue
        global depth_queue
        with lock:  
            # 如果队列未满，直接放入新数据  
            if len(color_queue) < color_queue.maxlen:  
                color_queue.append(rgb_image) 
            else:
                # 如果队列已满，移除最旧的元素  
                color_queue.popleft() 
                color_queue.append(rgb_image)
            
            if len(depth_queue) < depth_queue.maxlen:  
                depth_queue.append(depth_image)  
            else:
                # 如果队列已满，移除最旧的元素  
                depth_queue.popleft() 
                depth_queue.append(depth_image)
                
    def spin(self):  
        # 保持节点运行  
        rospy.spin()  