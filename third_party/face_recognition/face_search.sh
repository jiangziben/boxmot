#!/usr/bin/env bash

# 测试图片文件
image_dir="/home/jiangziben/data/people_tracking/zk" # 测试图片的目录
out_dir="output/"  # 保存检测结果
python face_search.py --image_dir $image_dir --database /home/jiangziben/CodeProject/boxmot/data/database-resnet50.json

# # 测试视频文件
# video_file="/home/jiangziben/data/people_tracking/gait/20241030_165151.mp4"
# python face_search.py --video_file $video_file --database /home/jiangziben/CodeProject/boxmot/data/known_people/database

# # 测试摄像头
# video_file="0"
# python face_search.py --video_file $video_file

