# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import argparse
import cv2
import numpy as np
from functools import partial
from pathlib import Path

import torch

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS,DATA
from boxmot.utils.checks import RequirementsChecker
from tracking.detectors import get_yolo_inferer

checker = RequirementsChecker()
checker.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git', ))  # install
import sys
import os
# 获取当前脚本路径
script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_path, "../third_party/ultralytics")) # add ultralytics to path

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.data.utils import VID_FORMATS
from ultralytics.utils.plotting import save_one_box
from ultralytics.engine.results import Results
from tracking.face_detect import FaceDetector,FaceDetectorV2,FaceDetectorV3,FaceDetectorV5
import time
from tracking.get_3d_pos import get_foot_point_from_bbox_and_depth,get_foot_point_from_keypoints_and_depth
import rospy
from geometry_msgs.msg import PoseStamped

import threading
from ultralytics.utils.node import ListenerNode
import rospy

# 相机内参矩阵 (fx, fy, cx, cy)
intrinsics = np.array([
    [1031.449707, 0, 957.330994],  # fx, 0, cx
    [0, 1031.363525, 534.711609],  # 0, fy, cy
    [0, 0, 1]           # 0, 0, 1
])
camera2world_extrinsic = np.array([[0,0,1,0],
                                   [-1,0,0,0],
                                   [0,-1,0,0],
                                   [0,0,0,1]])

class PersonInfo:
    def __init__(self, track_id, name_id,face_confidence):
        self.track_id = track_id
        self.name_id  = name_id
        self.face_confidence = face_confidence

class PeopleId:
    def __init__(self):
        self.people_id = {}

    def is_box_inside(self,boxA, boxB):
        """
        判断 boxA 是否完全在 boxB 内部。

        参数:
        - boxA: [x_min, y_min, x_max, y_max] 表示的第一个框
        - boxB: [x_min, y_min, x_max, y_max] 表示的第二个框
        
        返回:
        - True 如果 boxA 完全在 boxB 内部，否则 False
        """
        return (boxA[0] >= boxB[0] and boxA[1] >= boxB[1] and
                boxA[2] <= boxB[2] and boxA[3] <= boxB[3])
    
    def update(self,result,face_ids,face_confidences,person_indexes):
        for i,name_id in enumerate(face_ids):
            box = result.boxes.data[person_indexes[i]]
            track_id = int(box[4])
            person_info = PersonInfo(track_id, name_id,face_confidences[i])
            if name_id != "unknown" and (name_id not in self.people_id or person_info.face_confidence > self.people_id[name_id].face_confidence):
                self.people_id[name_id] = person_info
        
    def get_person_id(self,track_id):
        for name_id, person_info in self.people_id.items():
            if person_info.track_id == track_id:
                return name_id
        return -1
 
    def get_track_id(self,name_id):
        if name_id in self.people_id:
            return self.people_id[name_id].track_id  
        else:
            return -1
    
    def get_person_trajectory(self, name_id, active_tracks):
        track_id = self.get_track_id(name_id)
        for active_track in active_tracks:
            if 0 == int(active_track.cls): # only track people
                if active_track.id == track_id:
                    return active_track
        return None
    
    def get_keypoints(self,name_id,result):
        track_id = self.get_track_id(name_id)
        for i,box in enumerate(result.boxes.data):
            if track_id == int(box[4]): 
                return result.keypoints.data[i]                 
        return None

def on_predict_start(predictor, persist=False):
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    """

    assert predictor.custom_args.tracking_method in TRACKERS, \
        f"'{predictor.custom_args.tracking_method}' is not supported. Supported ones are {TRACKERS}"

    tracking_config = TRACKER_CONFIGS / (predictor.custom_args.tracking_method + '.yaml')
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.custom_args.tracking_method,
            tracking_config,
            predictor.custom_args.reid_model,
            predictor.device,
            predictor.custom_args.half,
            predictor.custom_args.per_class
        )
        # motion only modeles do not have
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers

def plot_results(self,results:Results, people_id:PeopleId, img: np.ndarray, show_trajectories: bool, thickness: int = 2, fontscale: float = 0.5) -> np.ndarray:
    """
    Visualizes the trajectories of all active tracks on the image. For each track,
    it draws the latest bounding box and the path of movement if the history of
    observations is longer than two. This helps in understanding the movement patterns
    of each tracked object.

    Parameters:
    - img (np.ndarray): The image array on which to draw the trajectories and bounding boxes.
    - show_trajectories (bool): Whether to show the trajectories.
    - thickness (int): The thickness of the bounding box.
    - fontscale (float): The font scale for the text.

    Returns:
    - np.ndarray: The image array with trajectories and bounding boxes of all active tracks.
    """
    for box in results.boxes.data:
        id = int(box[4])
        if -1 == people_id.get_person_id(id):
            name = "unknown"
        else:
            name = people_id.get_person_id(id)#people_id.known_people_names[people_id.get_person_id(a.id)] 
        img = cv2.putText(
            img,
            f'name: {name}',
            (int(box[0]), int(box[3]) + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontscale,
            self.id_to_color(id),
            thickness
        )  
    # for a in self.active_tracks:
    #     if a.history_observations:
    #         if len(a.history_observations) > 2:
    #             box = a.history_observations[-1]
    #             img = self.plot_box_on_img(img, box, a.conf, a.cls, a.id, thickness, fontscale)
    #             if 0 == a.cls:
    #                 if -1 == people_id.get_person_id(a.id):
    #                     name = "unknown"
    #                 else:
    #                     name = people_id.get_person_id(a.id)#people_id.known_people_names[people_id.get_person_id(a.id)] 
    #                 img = cv2.putText(
    #                     img,
    #                     f'name: {name}',
    #                     (int(box[0]), int(box[3]) + 10),
    #                     cv2.FONT_HERSHEY_SIMPLEX,
    #                     fontscale,
    #                     self.id_to_color(a.id),
    #                     thickness
    #                 )                    
    #             if show_trajectories:
    #                 img = self.plot_trackers_trajectories(img, a.history_observations, a.id)
            
    return img

def camera_to_world_coordinates(camera_points, rotation_matrix, translation_vector):
    """
    将相机坐标系下的点变换到世界坐标系。
    
    :param camera_points: 相机坐标系下的点数组 (N x 3)
    :param rotation_matrix: 相机到世界坐标系的旋转矩阵 (3x3)
    :param translation_vector: 相机到世界坐标系的平移向量 (3,)
    :return: 世界坐标系下的点数组 (N x 3)
    """
    # 确保输入的形状正确
    camera_points = np.asarray(camera_points)  # (N x 3)
    rotation_matrix = np.asarray(rotation_matrix)  # (3 x 3)
    translation_vector = np.asarray(translation_vector).reshape(1, 3)  # (1 x 3)
    
    # 应用世界坐标系变换: world_point = R * camera_point + T
    world_points = np.dot(camera_points, rotation_matrix.T) + translation_vector
    
    return world_points


@torch.no_grad()
def run(args):

    ul_models = ['yolov8', 'yolov9', 'yolov10', 'yolov11', 'rtdetr', 'sam']

    yolo = YOLO(
        args.yolo_model if any(yolo in str(args.yolo_model) for yolo in ul_models) else 'yolov8n.pt',
    )

    results = yolo.track(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        agnostic_nms=args.agnostic_nms,
        show=False,
        stream=True,
        device=args.device,
        show_conf=args.show_conf,
        save_txt=args.save_txt,
        show_labels=args.show_labels,
        save=args.save,
        verbose=args.verbose,
        exist_ok=args.exist_ok,
        project=args.project,
        name=args.name,
        classes=args.classes,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
        line_width=args.line_width
    )

    yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))
    # yolo.add_callback('on_predict_postprocess_end', partial(on_predict_postprocess_end,intrinsics=intrinsics))

    if not any(yolo in str(args.yolo_model) for yolo in ul_models):
        # replace yolov8 model
        m = get_yolo_inferer(args.yolo_model)
        model = m(
            model=args.yolo_model,
            device=yolo.predictor.device,
            args=yolo.predictor.args
        )
        yolo.predictor.model = model

    # store custom args in predictor
    yolo.predictor.custom_args = args
    face_detector = FaceDetectorV5(args.yolo_model,args.reid_model, args.host_image_path)
    people_id = PeopleId()
    host_detected = False

    # 初始化ROS节点
    # rospy.init_node('people_tracking_node', anonymous=True)
    
    # 创建一个PoseStamped消息
    pose_msg = PoseStamped()
    # 创建一个发布者，发布到 /float_array 话题上，消息类型为 PoseStamped
    pub = rospy.Publisher('/host_foot_point', PoseStamped, queue_size=10)
    
    # 设置发布的频率（10Hz）
    rate = rospy.Rate(10)  # 10 Hz
    host_name = args.host_name
    host_id = host_name#(face_detector.known_face_names == host_name).argmax()
    if args.show:
        cv2.namedWindow("PeopleTracking",cv2.WINDOW_NORMAL)
    face_ids,face_confidences,person_indexes=[],[],[]
    for i,r in enumerate(results):
        time_start = time.time()
        if rospy.is_shutdown():
            break  
        if not host_detected:
            # face_ids,face_locations,_,face_confidences,person_indexes = face_detector.detect_faces(r.orig_img,r)
            face_infos = face_detector.detect_faces(r.orig_img,r)
            if face_infos and len(face_infos['label'])>0:
                face_ids = face_infos['label']
                face_confidences = face_infos['score']
                person_indexes = face_infos['person_index']
                people_id.update(r, face_ids,face_confidences,person_indexes)
            
        # host_trajectory = people_id.get_person_trajectory(host_id,yolo.predictor.trackers[0].active_tracks)
        keypoints = people_id.get_keypoints(host_id,r)
        if keypoints is not None:
            host_detected = True
            depth = yolo.predictor.batch[3] if len(yolo.predictor.batch) == 4 else None
            foot_point = get_foot_point_from_keypoints_and_depth(keypoints,depth[0],intrinsics,scale=1000.0)
            if foot_point is not None:
                # print(f"落脚点的 3D 坐标：X={foot_point[0]}, Y={foot_point[1]}, Z={foot_point[2]}")
                # 更新消息中的数据
                foot_point_w = foot_point
                # foot_point_w = camera_to_world_coordinates(foot_point,camera2world_extrinsic[0:3,0:3], camera2world_extrinsic[0:3,3])[0]
# 
                # 设置位置 (x, y, z)
                pose_msg.pose.position.x = foot_point_w[0]
                pose_msg.pose.position.y = foot_point_w[1]
                pose_msg.pose.position.z = foot_point_w[2]

                # 设置方向 (以四元数表示)
                pose_msg.pose.orientation.x = 0.0
                pose_msg.pose.orientation.y = 0.0
                pose_msg.pose.orientation.z = 0.0
                pose_msg.pose.orientation.w = 1.0

            else:
                print("无法获取落脚点的 3D 坐标，可能是深度无效。")                    
        else:
            host_detected = False
            # 设置位置 (x, y, z)
            pose_msg.pose.position.x = 0
            pose_msg.pose.position.y = 0
            pose_msg.pose.position.z = 0

            # 设置方向 (以四元数表示)
            pose_msg.pose.orientation.x = 0.0
            pose_msg.pose.orientation.y = 0.0
            pose_msg.pose.orientation.z = 0.0
            pose_msg.pose.orientation.w = 1.0

        # 填充header信息
        pose_msg.header.stamp = rospy.Time.now()  
        pose_msg.header.frame_id = "dog"  # 设置坐标系frame_id
        # 发布消息到话题
        pub.publish(pose_msg)
        # 打印日志，方便调试
        rospy.loginfo("Publishing: %s,%s,%s", pose_msg.pose.position.x,pose_msg.pose.position.y,pose_msg.pose.position.z)
        print("used time: ", time.time()-time_start)
        if args.show is True:
            img_plot = r.plot(boxes=True, masks=False, labels=True) 
            img = plot_results(yolo.predictor.trackers[0],r,people_id,img_plot, args.show_trajectories,fontscale=1.0,thickness=5)
            cv2.imshow('PeopleTracking', img)     
            key = cv2.waitKey(40) & 0xFF
            if key == ord(' ') or key == ord('q'):
                break
        # 按照10Hz的频率发布
        rate.sleep()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / "yolov11m-pose.pt", #WEIGHTS / 'yolov8n',
                        help='yolo model path')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.engine',
                        help='reid model path')
    parser.add_argument('--tracking-method', type=str, default='botsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack, imprassoc')
    # parser.add_argument('--source', type=str, default='/home/jiangziben/data/people_tracking/3d/follow/',#'0',
    #                     help='file/dir/URL/glob, 0 for webcam')
    # parser.add_argument('--source', type=str, default='6',
    #                     help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--source', type=str, default=DATA / "follow_simple/",
                        help='file/dir/URL/glob/ros, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.4,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--save', action='store_true',
                        help='save video tracking results')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--show-labels', action='store_false',
                        help='either show all or only bboxes')
    parser.add_argument('--show-conf', action='store_false',
                        help='hide confidences when show')
    parser.add_argument('--show-trajectories', action='store_true',
                        help='show confidences')
    parser.add_argument('--save-txt', action='store_true',
                        help='save tracking results in a txt file')
    parser.add_argument('--save-id-crops', action='store_true',
                        help='save each crop to its respective id folder')
    parser.add_argument('--line-width', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--per-class', default=False, action='store_true',
                        help='not mix up classes when tracking')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='print results per frame')
    parser.add_argument('--agnostic-nms', default=False, action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--host_image_path', type=str, default= DATA / "known_people/",
                        help='host image path')
    parser.add_argument('--host_name', type=str, default= "jzb",
                        help='host name')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    rospy.init_node('tracking', anonymous=True)
    opt = parse_opt()
    if opt.source != 'ros':   
        run(opt)
    else:
        try: 
            run_thread = threading.Thread(target=run, args=(opt,)) 
            run_thread.start()              
            listener = ListenerNode()  
            listener.spin()  
        except KeyboardInterrupt:
            pass
        print("Exiting Program")