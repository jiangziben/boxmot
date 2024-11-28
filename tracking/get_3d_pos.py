# import open3d as o3d
# import numpy as np
# import cv2

# def extract_depth_roi(depth_map, bbox):
#     """
#     提取深度图中边界框区域的深度信息
    
#     :param depth_map: 输入的深度图 (numpy array)
#     :param bbox: 边界框 (x_min, y_min, x_max, y_max)
#     :return: ROI 区域的深度图
#     """
#     x_min, y_min, x_max, y_max = bbox

#     # 确保边界框不超出深度图范围
#     x_min = max(0, int(x_min))
#     y_min = max(0, int(y_min))
#     x_max = min(depth_map.shape[1], int(x_max))
#     y_max = min(depth_map.shape[0], int(y_max))

#     # 提取边界框内的深度图区域
#     depth_roi = np.zeros_like(depth_map)
#     depth_roi[y_min:y_max, x_min:x_max] = depth_map[y_min:y_max, x_min:x_max]
#     return depth_roi

# def depth_to_point_cloud(depth_roi, intrinsics, depth_scale=1000.0, depth_trunc=3.0):
#     """
#     将深度图区域转换为点云
    
#     :param depth_roi: ROI 区域的深度图
#     :param intrinsics: 相机内参 (open3d.camera.PinholeCameraIntrinsic)
#     :param depth_scale: 深度值缩放比例
#     :param depth_trunc: 最大深度裁剪
#     :return: Open3D 点云对象
#     """
#     # 将 numpy 深度图转换为 Open3D 图像格式
#     depth_image = o3d.geometry.Image(depth_roi.astype(np.uint16))

#     # 将深度图转换为点云
#     point_cloud = o3d.geometry.PointCloud.create_from_depth_image(
#         depth_image, 
#         intrinsics, 
#         depth_scale=depth_scale, 
#         depth_trunc=depth_trunc,
#         stride=1
#     )
    
#     return point_cloud

# def main():
#     # 示例深度图加载
#     depth_image = cv2.imread("/home/jiangziben/data/people_tracking/3d/1/Depth/007756_0032_377901688_1730893837934381_Depth_1280x720.png",cv2.IMREAD_UNCHANGED)  # 或者使用 np.load('depth_map.npy') 然后用 open3d 读取

#     # 定义边界框 (示例框)
#     # bbox = (370, 54, 523, 441)  # 定义感兴趣区域 (x_min, y_min, x_max, y_max)
#     center_pos = (452,247)
#     box_size = 20
#     bbox = (int(center_pos[0]-box_size/2), int(center_pos[1]-box_size/2), int(center_pos[0]+box_size/2), int(center_pos[1]+box_size/2))
#     x_min, y_min, x_max, y_max = bbox
#     depth_mean = np.mean(depth_image[y_min:y_max, x_min:x_max])
#     print("depth_mean:",depth_mean)
#     # 提取边界框内的深度图
#     depth_roi = extract_depth_roi(depth_image, bbox)

#     # 定义相机内参 (根据相机参数设置，示例为常见参数)
#     intrinsics = o3d.camera.PinholeCameraIntrinsic()
#     intrinsics.set_intrinsics(width=1280, height=720, fx=687.633179, fy=687.575684, cx=638.220703, cy=356.474426)
    
#     # 将深度图区域转换为点云
#     point_cloud = depth_to_point_cloud(depth_roi, intrinsics, depth_scale=1000.0, depth_trunc=20.0)
    
#     # 显示生成的点云
#     o3d.visualization.draw_geometries([point_cloud])

#     # 保存点云到文件
#     o3d.io.write_point_cloud("roi_point_cloud.pcd", point_cloud)
#     print("ROI 点云已保存为 'roi_point_cloud.pcd'")

# if __name__ == "__main__":
#     main()


import cv2
import numpy as np
import os
from ultralytics import YOLO
def pixel_to_camera_coordinates(x, y, depth, intrinsics):
    """
    将像素坐标 (x, y) 和深度值转换为相机坐标系下的 3D 坐标。
    
    :param x: 像素坐标 x
    :param y: 像素坐标 y
    :param depth: 深度值
    :param intrinsics: 相机内参矩阵 (3x3)
    :return: 相机坐标系下的 3D 坐标 (X, Y, Z)
    """
    camera_xyz = np.zeros(3)
    # 使用相机内参将像素坐标转换为相机坐标系下的 3D 坐标
    fx = intrinsics[0, 0]  # 焦距 (fx)
    fy = intrinsics[1, 1]  # 焦距 (fy)
    cx = intrinsics[0, 2]  # 光心 x 坐标 (cx)
    cy = intrinsics[1, 2]  # 光心 y 坐标 (cy)

    # 计算相机坐标系下的 3D 坐标
    camera_xyz[2] = depth  # 深度值
    camera_xyz[0] = (x - cx) * camera_xyz[2] / fx
    camera_xyz[1] = (y - cy) * camera_xyz[2] / fy

    return camera_xyz

def get_foot_point_from_bbox_and_depth(bbox, depth_map, intrinsics, scale=1.0):
    """
    从给定的边界框 (bbox) 和深度图获取落脚点的相机坐标系下的 3D 坐标。
    
    :param bbox: 边界框 (x_min, y_min, x_max, y_max)
    :param depth_map: 深度图 (numpy array)
    :param intrinsics: 相机内参矩阵 (3x3)
    :return: 落脚点的 3D 坐标 (X, Y, Z)
    """
    x_min, y_min, x_max, y_max = bbox

    # 选择边界框底部区域的中点作为落脚点
    x_foot = (x_min + x_max) // 2
    y_foot = y_max  # 选择边界框的底部

    # 获取深度图中对应像素的深度值
    bbox_center = ((bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2)
    roi_size = 10
    depth_foot = np.mean(depth_map[int(bbox_center[1]-roi_size/2):int(bbox_center[1]+roi_size/2), int(bbox_center[0]-roi_size/2):int(bbox_center[0]+roi_size/2)])/scale#depth_map[y_foot, x_foot]/1000.0

    # 如果深度值有效（大于零）
    if depth_foot > 0:
        # 将像素坐标和深度值转换为相机坐标系下的 3D 坐标
        X, Y, Z = pixel_to_camera_coordinates(x_foot, y_foot, depth_foot, intrinsics)
        return X, Y, Z
    else:
        # 如果深度值无效，则返回 None
        return None

def get_pos_from_keypoint(keypoint,depth_map, intrinsics, scale=1.0):
    # 获取深度图中对应像素的深度值
    roi_size = 20
    if keypoint[2] > 0.5:
        region_depth = depth_map[int(keypoint[1]-roi_size/2):int(keypoint[1]+roi_size/2), int(keypoint[0]-roi_size/2):int(keypoint[0]+roi_size/2)]/scale
        depth_foot = np.median(region_depth)
        # depth_foot = np.mean(depth_map[int(keypoint[1]-roi_size/2):int(keypoint[1]+roi_size/2), int(keypoint[0]-roi_size/2):int(keypoint[0]+roi_size/2)])/scale#depth_map[y_foot, x_foot]/1000.0
    else:
        return None
    # 如果深度值有效（大于零）
    if depth_foot > 0:
        # 将像素坐标和深度值转换为相机坐标系下的 3D 坐标
        xyz_camera = pixel_to_camera_coordinates(keypoint[0], keypoint[1], depth_foot, intrinsics)
        return xyz_camera
    else:
        # 如果深度值无效，则返回 None
        return None

def get_foot_point_from_keypoints_and_depth(keypoints, depth_map, intrinsics, scale=1.0):
    left_foot_index = 15  #
    right_foot_index = 16  #  
    keypoints_array = keypoints.cpu().numpy() 
    if keypoints_array.shape[0] == 0:
        return None
    left_foot_keypoint = keypoints_array[left_foot_index,:]
    right_foot_keypoint = keypoints_array[right_foot_index,:]
    left_foot_pos = get_pos_from_keypoint(left_foot_keypoint,depth_map,intrinsics, scale)
    right_foot_pos = get_pos_from_keypoint(right_foot_keypoint,depth_map,intrinsics, scale)
    foot_pos = None
    if left_foot_pos is not None and right_foot_pos is not None:
        foot_pos = (left_foot_pos + right_foot_pos) / 2.0
    elif left_foot_pos is not None:
        foot_pos = left_foot_pos
    elif right_foot_pos is not None:
        foot_pos = right_foot_pos
    return foot_pos


def get_images_in_folder(folder_path):
    """
    获取指定文件夹下所有图片文件
    :param folder_path: 文件夹路径
    :return: 图片文件列表
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']  # 支持的图片格式
    images = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                images.append(os.path.join(root, file))

    return images 

if __name__ == "__main__":
    # 示例：加载图像和深度图
    folder_path = "/home/jiangziben/data/people_tracking/3d/1/"
    images_path = get_images_in_folder(os.path.join(folder_path, "Color"))
    depth_images_path = get_images_in_folder(os.path.join(folder_path, "Depth"))

    # 相机内参矩阵 (fx, fy, cx, cy)
    intrinsics = np.array([
        [687.633179, 0, 638.220703],  # fx, 0, cx
        [0, 687.575684, 356.474426],  # 0, fy, cy
        [0, 0, 1]           # 0, 0, 1
    ])

    yolo_model = YOLO("tracking/weights/yolov11m-pose.pt")  # build from YAML and transfer weights
    for image_path,depth_map_path in zip(images_path,depth_images_path):
        image = cv2.imread(image_path)  # 2D 图像
        depth_map = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED)  # 深度图
        # boxes = yolo_model.predict(image,verbose=False)[0].boxes.data
        # for box in boxes:
        #     x1, y1, x2, y2, conf, cls = box.cpu().numpy()
        #     if cls == 0:         
        #         # 获取落脚点的 3D 坐标
        #         foot_point = get_foot_point_from_keypoints_and_depth((x1,y1,x2,y2), depth_map, intrinsics)

        #         if foot_point:
        #             print(f"落脚点的 3D 坐标：X={foot_point[0]}, Y={foot_point[1]}, Z={foot_point[2]}")
        #         else:
        #             print("无法获取落脚点的 3D 坐标，可能是深度无效。")
        keypoints = yolo_model.predict(image,verbose=False)[0].keypoints.data
        for keypoint in keypoints:
            # 获取落脚点的 3D 坐标
            foot_point = get_foot_point_from_keypoints_and_depth(keypoint, depth_map, intrinsics)

            if foot_point:
                print(f"落脚点的 3D 坐标：X={foot_point[0]}, Y={foot_point[1]}, Z={foot_point[2]}")
            else:
                print("无法获取落脚点的 3D 坐标，可能是深度无效。")