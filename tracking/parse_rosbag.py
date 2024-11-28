#!/usr/bin/env python3

import os
import cv2
import rosbag
from cv_bridge import CvBridge

import argparse
import os
import numpy as np
def main():
    # 创建解析器
    parser = argparse.ArgumentParser(description='参数解析示例')

    # 添加参数
    parser.add_argument('-i', '--input_file', type=str,default="/home/jiangziben/data/people_tracking/3d/2024-11-27-15-27-04.bag", help='输入文件路径')
    parser.add_argument('-o', '--output_folder', type=str,default="/home/jiangziben/data/people_tracking/3d/2024-11-27-15-27-04",help='输出文件夹路径')

    # 解析参数
    args = parser.parse_args()
    # 初始化 CvBridge
    bridge = CvBridge()

    # 设置保存图像的目录
    output_dir = args.output_folder
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    color_image_path = os.path.join(output_dir, 'Color')
    if not os.path.exists(color_image_path):
        os.makedirs(color_image_path)
    depth_image_path = os.path.join(output_dir, 'Depth')
    if not os.path.exists(depth_image_path):
        os.makedirs(depth_image_path)
    fisheye_image_path = os.path.join(output_dir, 'Fisheye')
    if not os.path.exists(fisheye_image_path):
        os.makedirs(fisheye_image_path)
    # 打开 rosbag 文件
    bag = rosbag.Bag(args.input_file, 'r')

    # 选择要读取的图像话题
    image_topic = '/camera/color/image_raw/compressed'
    depth_image_topic = '/camera/depth/image_raw'
    fisheye_image_topic = '/cam_front/csi_cam/image_raw/compressed'
    # 解析 rosbag 文件中的图像信息
    for topic, msg, t in bag.read_messages(topics=[image_topic, depth_image_topic,fisheye_image_topic]):
        if topic[-1] == '/':
            topic = topic[:-1]
        # 将 ROS Image 消息转换为 OpenCV 格式的图像
        try:
            if topic == image_topic:
                cv_image = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            elif topic == depth_image_topic:
                #  # 解码 CompressedImage 数据
                cv_depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            elif topic == fisheye_image_topic:
                 # 解码 CompressedImage 数据
                np_arr = np.frombuffer(msg.data, np.uint8)
                cv_fisheye_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)                
        except Exception as e:
            print(f"Error converting image: {e}")
            continue
        
        # 生成图像文件名，使用时间戳作为名称
        timestamp = str(t.to_nsec())  # 转换时间戳为纳秒
        image_filename = os.path.join(color_image_path, f"frame_{timestamp}.png")
        depth_image_filename = os.path.join(depth_image_path, f"frame_{timestamp}.png")
        fisheye_image_filename = os.path.join(fisheye_image_path, f"frame_{timestamp}.png")

        # 保存图像文件
        if topic == image_topic:
            cv2.imwrite(image_filename, cv_image)
            print(f"Saved image: {image_filename}")
        elif topic == depth_image_topic:   
            cv2.imwrite(depth_image_filename, cv_depth_image)
            print(f"Saved depth image: {depth_image_filename}")
        elif topic == fisheye_image_topic:
            cv2.imwrite(fisheye_image_filename, cv_fisheye_image)
            print(f"Saved image: {fisheye_image_filename}")
    # 关闭 bag 文件
    bag.close()


if __name__ == '__main__':
    main()


