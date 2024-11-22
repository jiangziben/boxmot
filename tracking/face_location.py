import cv2
import numpy as np
import torch
def calculate_eye_distance(keypoints, left_eye_index, right_eye_index):
    """
    计算两个眼睛之间的距离。

    参数：
        keypoints (list of list): 关键点数据，格式为 [x, y, confidence]。
        left_eye_index (int): 左眼的关键点索引。
        right_eye_index (int): 右眼的关键点索引。

    返回：
        float: 左眼和右眼之间的距离。
    """
    left_eye = keypoints[left_eye_index][:2]  # 提取左眼的 (x, y) 坐标
    right_eye = keypoints[right_eye_index][:2]  # 提取右眼的 (x, y) 坐标
    return np.linalg.norm(left_eye - right_eye)  # 返回欧几里得距离

def get_face_location_from_keypoints(keypoints, image_shape,confidence_threshold=0.5, u_padding_ratio=1.1, v_padding_ratio=1.4):
    """
    根据关键点和眼睛距离计算脸部位置。

    参数：
        keypoints (list of list): 关键点数据，格式为 [x, y, confidence]。
        eye_distance (float): 左眼和右眼之间的距离。
            
    """  
    # 与脸部相关的关键点索引
    face_indices = [0, 1, 2]  # Nose, Left Eye, Right Eye, Left Ear, Right Ear
    left_eye_index = 1  # 左眼的索引
    right_eye_index = 2  # 右眼的索引  
    img_height, img_width = image_shape[:2]
  
    keypoints_array = keypoints.cpu().numpy()  
    if keypoints_array.shape[0] == 0:
        return None
    # 检查所有关键点的置信度是否超过阈值
    confidences = keypoints_array[face_indices, 2]  # 提取置信度
    if np.any(confidences < confidence_threshold):
        return None

    # 计算眼睛之间的距离并动态计算 padding
    eye_distance = calculate_eye_distance(keypoints_array, left_eye_index, right_eye_index)
    u_padding = eye_distance * u_padding_ratio  # 使用眼睛距离作为 padding
    v_padding = eye_distance * v_padding_ratio  # 使用眼睛距离作为 padding
    # 提取脸部相关的关键点坐标
    nose_x,nose_y = keypoints_array[0, :2]  # 提取 (x, y)

    # 添加 padding
    x_min = nose_x - u_padding
    y_min = nose_y - v_padding
    x_max = nose_x + u_padding
    y_max = nose_y + v_padding
    # 限制边界框在图像范围内
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_width, x_max)
    y_max = min(img_height, y_max)
    # 存储结果
    face_bbox = np.stack([x_min, y_min, x_max, y_max])
    return face_bbox


def calculate_face_bboxes_with_dynamic_padding(keypoints_list,image_shape, confidence_threshold=0.5, u_padding_ratio=1.1,v_padding_ratio=1.6):
    """
    计算人脸边界框，padding 自动根据眼睛之间的距离膨胀。

    参数：
        keypoints_list (list of list of lists): YOLOv8 Pose 输出的关键点数据，每个人是一组关键点。

        confidence_threshold (float, optional): 关键点的置信度阈值，默认为 0.5。
        u_padding_ratio (float, optional): 根据眼睛距离计算u方向 padding 的比例，默认为 1.0。
        v_padding_ratio (float, optional): 根据眼睛距离计算v方向 padding 的比例，默认为 1.0。
    返回：
        list of lists: 每个人的脸部边界框，格式为 [[x_min, y_min, x_max, y_max], ...]。
    """

    face_bboxes = []

    for keypoints in keypoints_list:
        face_bbox = get_face_location_from_keypoints(keypoints=keypoints, image_shape=image_shape,confidence_threshold=confidence_threshold, u_padding_ratio=u_padding_ratio, v_padding_ratio=v_padding_ratio)
        face_bboxes.append(face_bbox)
    return face_bboxes

def draw_face_bboxes(image, face_bboxes, color=(0, 255, 0), thickness=2, show_index=True):
    """
    在图像上绘制脸部边界框。

    参数：
        image (numpy.ndarray): 输入的图像，格式为 OpenCV 的 BGR 图像。
        face_bboxes (list of lists): 每个人的脸部边界框，格式为 [[x_min, y_min, x_max, y_max], ...]。
        color (tuple, optional): 边界框的颜色，默认为绿色 (0, 255, 0)。
        thickness (int, optional): 边界框的线条粗细，默认为 2。
        show_index (bool, optional): 是否显示边界框的索引号，默认为 True。

    返回：
        numpy.ndarray: 带有边界框的图像。
    """
    for i, bbox in enumerate(face_bboxes):
        if bbox is not None:
            x_min, y_min, x_max, y_max = map(int, bbox)  # 将坐标转换为整数
            # 绘制边界框
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
            
            if show_index:
                # 在边界框左上角显示索引号
                label = f"Face {i+1}"
                font_scale = 0.5
                font_thickness = 1
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                text_x = x_min
                text_y = y_min - 5 if y_min - 5 > 0 else y_min + text_size[1] + 5
                cv2.putText(
                    image, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness, cv2.LINE_AA
                )
    return image


from ultralytics import YOLO

if __name__ == '__main__':
    # Configure the tracking parameters and run the tracker
    model = YOLO("/home/jiangziben/CodeProject/boxmot/tracking/weights/yolov11m-pose.pt")
    results = model.predict(source="/home/jiangziben/data/people_tracking/multi_people", conf=0.3, iou=0.5,stream=True)
    # 计算脸部边界框（置信度阈值为 0.5，padding 为 10）
    confidence_threshold = 0.5

    for result in results:
        # print(result)
        # YOLOv8 输出关键点数据
        keypoints_list = result.keypoints.data

        # 计算脸部边界框，padding 根据眼睛的距离自动计算
        face_bboxes = calculate_face_bboxes_with_dynamic_padding(
            keypoints_list,result.orig_img.shape, confidence_threshold=confidence_threshold, u_padding_ratio=1.1,v_padding_ratio=1.6,
        ) 
    # 绘制边界框
        output_image = draw_face_bboxes(result.orig_img, face_bboxes, color=(255, 0, 0), thickness=2)
        # 打印结果
        print("Face Bounding Boxes:", face_bboxes)
        # 显示图像
        cv2.imshow("Face Bounding Boxes", output_image)
        cv2.waitKey(40)
    cv2.destroyAllWindows()    
