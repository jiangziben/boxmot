import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from collections import deque
import threading

color_queue = deque(maxlen=5)
lock = threading.Lock()

class ListenerNode:  
    def __init__(self):  
        # Subscribe to the color image topic  
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)

    def image_callback(self, msg):  
            # Convert ROS Image message to OpenCV image  
        bridge = CvBridge()  
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8') 
        # 回调函数，处理接收到的消息  
        global color_queue

        with lock:  
            # 如果队列未满，直接放入新数据  
            if len(color_queue) < color_queue.maxlen:  
                color_queue.append(cv_image)  
            else:
                # 如果队列已满，移除最旧的元素  
                color_queue.popleft()  
                color_queue.append(cv_image)

            
    def spin(self):  
        # 保持节点运行  
        rospy.spin()  