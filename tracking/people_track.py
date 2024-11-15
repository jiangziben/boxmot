# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import argparse
import cv2
import numpy as np
from functools import partial
from pathlib import Path

import torch

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS
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
from tracking.face_detect import FaceDetector
import time
from tracking.get_3d_pos import get_foot_point_from_bbox_and_depth
import rospy
from geometry_msgs.msg import PoseStamped

class PersonInfo:
    def __init__(self, track_id, name_id, face_box):
        self.track_id = track_id
        self.name_id  = name_id
        self.face_box = face_box

class PeopleId:
    def __init__(self,known_people_names):
        self.people_id = {}
        self.known_people_names = known_people_names

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
    
    def init(self,active_tracks, face_names, face_locations):
        for active_track in active_tracks:
            if 0 == int(active_track.cls): # only track people
                if active_track.history_observations:
                    if len(active_track.history_observations) > 2:
                        box = active_track.history_observations[-1]
                        person_info = PersonInfo(active_track.id, -1, box)
                        for face_location in face_locations:
                            if self.is_box_inside(face_location, box[0:4]):
                                name_id = face_names[face_locations.index(face_location)]
                                person_info.name_id = name_id
                                self.people_id[active_track.id] = person_info
                                return True
        return False
    def get_person_id(self,track_id):
        if track_id in self.people_id:
            return self.people_id[track_id].name_id  
        else:
            return -1
    def get_track_id(self,name_id):
        for track_id, person_info in self.people_id.items():
            if person_info.name_id == name_id:
                return track_id
        return -1
    
    def get_person_trajectory(self, name_id, active_tracks):
        track_id = self.get_track_id(name_id)
        for active_track in active_tracks:
            if 0 == int(active_track.cls): # only track people
                if active_track.id == track_id:
                    return active_track
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

# def on_predict_postprocess_end(predictor: object, persist: bool = False, intrinsics: np.ndarray = None) -> None:
#     """
#     Postprocess detected boxes and update with object tracking.

#     Args:
#         predictor (object): The predictor object containing the predictions.
#         persist (bool): Whether to persist the trackers if they already exist.

#     Examples:
#         Postprocess predictions and update with tracking
#         >>> predictor = YourPredictorClass()
#         >>> on_predict_postprocess_end(predictor, persist=True)
#     """
#     depths = predictor.batch[3] if len(predictor.batch) == 4 else None

#     is_obb = predictor.args.task == "obb"
#     is_stream = predictor.dataset.mode == "stream"
#     for i in range(len(depths)):
#         det = (predictor.results[i].obb if is_obb else predictor.results[i].boxes.data).cpu().numpy()
#         if len(det) == 0:
#             continue
#         # predictor.results[i] = predictor.results[i]

#         # update_args = {"obb" if is_obb else "boxes": torch.as_tensor(tracks[:, :-1])}
#         # predictor.results[i].update(**update_args)

#         # boxes = predictor.results[i].boxes.data
#         # for box in boxes:
#         #     x1, y1, x2, y2,id, conf, cls = box.cpu().numpy()
#         #     if cls == 0:         
#         #         # 获取落脚点的 3D 坐标
#         #         foot_point = get_foot_point_from_bbox_and_depth((x1,y1,x2,y2), depths[i], intrinsics)

#         #         if foot_point:
#         #             print(f"落脚点的 3D 坐标：X={foot_point[0]}, Y={foot_point[1]}, Z={foot_point[2]}")
#         #             predictor.results[i] = predictor.results[i][idx]

#         #             update_args = {"pos_3d": torch.as_tensor(tracks[:, :-1])}
#         #             predictor.results[i].update(**update_args)
#         #         else:
#         #             print("无法获取落脚点的 3D 坐标，可能是深度无效。")    
#     #     tracker = predictor.trackers[i if is_stream else 0]
#     #     vid_path = predictor.save_dir / Path(path[i]).name
#     #     if not persist and predictor.vid_path[i if is_stream else 0] != vid_path:
#     #         predictor.vid_path[i if is_stream else 0] = vid_path

#     #     det = (predictor.results[i].obb if is_obb else predictor.results[i].boxes.data).cpu().numpy()
#     #     if len(det) == 0:
#     #         continue
#     #     tracks = tracker.update(det, im0s[i])
#     #     if len(tracks) == 0:
#     #         continue
#     #     idx = tracks[:, -1].astype(int)
#     #     predictor.results[i] = predictor.results[i][idx]

#     #     update_args = {"obb" if is_obb else "boxes": torch.as_tensor(tracks[:, :-1])}
#     #     predictor.results[i].update(**update_args)

def plot_results(self, people_id:PeopleId, img: np.ndarray, show_trajectories: bool, thickness: int = 2, fontscale: float = 0.5) -> np.ndarray:
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
    for a in self.active_tracks:
        if a.history_observations:
            if len(a.history_observations) > 2:
                box = a.history_observations[-1]
                img = self.plot_box_on_img(img, box, a.conf, a.cls, a.id, thickness, fontscale)
                if 0 == a.cls:
                    if -1 == people_id.get_person_id(a.id):
                        name = "unknown"
                    else:
                        name = people_id.known_people_names[people_id.get_person_id(a.id)] 
                    img = cv2.putText(
                        img,
                        f'name: {name}',
                        (int(box[0]), int(box[3]) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontscale,
                        self.id_to_color(a.id),
                        thickness
                    )                    
                if show_trajectories:
                    img = self.plot_trackers_trajectories(img, a.history_observations, a.id)
            
    return img


@torch.no_grad()
def run(args):
    # 相机内参矩阵 (fx, fy, cx, cy)
    intrinsics = np.array([
        [687.633179, 0, 638.220703],  # fx, 0, cx
        [0, 687.575684, 356.474426],  # 0, fy, cy
        [0, 0, 1]           # 0, 0, 1
    ])
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
    face_detector = FaceDetector(args.host_image_path)
    people_id = PeopleId(face_detector.known_face_names)
    face_detected = False
    host_id = 0
    # 初始化ROS节点
    rospy.init_node('people_tracking_node', anonymous=True)
    
    # 创建一个PoseStamped消息
    pose_msg = PoseStamped()
    # 创建一个发布者，发布到 /float_array 话题上，消息类型为 PoseStamped
    pub = rospy.Publisher('/host_foot_point', PoseStamped, queue_size=10)
    
    # 设置发布的频率（10Hz）
    rate = rospy.Rate(10)  # 10 Hz

    for i,r in enumerate(results):
        if rospy.is_shutdown():
            break      
        if not face_detected:
            face_ids,face_locations,_ = face_detector.detect_faces(r.orig_img)

            if len(face_ids) > 0 and host_id in face_ids:
                flag = people_id.init(yolo.predictor.trackers[0].active_tracks, face_ids, face_locations)
                if flag:
                    face_detected = True
        else:
            host_trajectory = people_id.get_person_trajectory(host_id,yolo.predictor.trackers[0].active_tracks)
            if host_trajectory is not None and host_trajectory.history_observations is not None:
                box = host_trajectory.history_observations[-1]
                depth = yolo.predictor.batch[3] if len(yolo.predictor.batch) == 4 else None
                foot_point = get_foot_point_from_bbox_and_depth(box,depth[0],intrinsics,scale=1000.0)
                if foot_point:
                    # print(f"落脚点的 3D 坐标：X={foot_point[0]}, Y={foot_point[1]}, Z={foot_point[2]}")
                    # 更新消息中的数据

                    pose_msg.header.frame_id = "dog"  # 设置坐标系frame_id

                    # 设置位置 (x, y, z)
                    pose_msg.pose.position.x = foot_point[0]
                    pose_msg.pose.position.y = foot_point[1]
                    pose_msg.pose.position.z = foot_point[2]

                    # 设置方向 (以四元数表示)
                    pose_msg.pose.orientation.x = 0.0
                    pose_msg.pose.orientation.y = 0.0
                    pose_msg.pose.orientation.z = 0.0
                    pose_msg.pose.orientation.w = 1.0

                else:
                    print("无法获取落脚点的 3D 坐标，可能是深度无效。")                    

        # 填充header信息
        pose_msg.header.stamp = rospy.Time.now()        
        # 发布消息到话题
        pub.publish(pose_msg)
        # 打印日志，方便调试
        rospy.loginfo("Publishing: %s,%s,%s", pose_msg.pose.position.x,pose_msg.pose.position.y,pose_msg.pose.position.z)
        
        if args.show is True:
            img = plot_results(yolo.predictor.trackers[0],people_id,r.orig_img, args.show_trajectories)
            cv2.imshow('BoxMOT', img)     
            key = cv2.waitKey(100) & 0xFF
            if key == ord(' ') or key == ord('q'):
                break
        # 按照10Hz的频率发布
        rate.sleep()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / "yolov11m_best.engine", #WEIGHTS / 'yolov8n',
                        help='yolo model path')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt',
                        help='reid model path')
    parser.add_argument('--tracking-method', type=str, default='botsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack, imprassoc')
    parser.add_argument('--source', type=str, default='/home/jiangziben/data/people_tracking/3d/follow/',#'0',
                        help='file/dir/URL/glob, 0 for webcam')
    # parser.add_argument('--source', type=str, default='6',
    #                     help='file/dir/URL/glob, 0 for webcam')
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
    parser.add_argument('--verbose', default=True, action='store_true',
                        help='print results per frame')
    parser.add_argument('--agnostic-nms', default=False, action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--host_image_path', type=str, default="/home/jiangziben/data/people_tracking/known_people/zk/zk.png",
                        help='host image path')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(opt)
