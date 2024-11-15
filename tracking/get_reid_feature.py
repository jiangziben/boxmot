#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2021/1/27
# @Author : zengwb


import os
import numpy as np
import cv2
import argparse
import time
from torch.backends import cudnn
import sys
sys.path.append('..')
import math
from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
from pathlib import Path
def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument('--weights', type=Path, default="/home/jiangziben/CodeProject/boxmot/tracking/weights/osnet_x0_25_msmt17.pt",
                        help='reid model path')
    parser.add_argument(
        "--input",
        # nargs="+",
        type=Path,
        default='/home/jiangziben/CodeProject/boxmot/datasets/person_bank/zk',
        help="path to known peopel image folders",
    )
    parser.add_argument(
        "--query",
        type=Path,
        default='/home/jiangziben/CodeProject/boxmot/datasets/person_bank',
        help="path to known peopel image folders",
    )
    parser.add_argument('--device', default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')

    return parser


class Reid_feature():
    def __init__(self,weights,device="0",half=True):
        self.model = ReidAutoBackend(
        weights=Path(weights), device=device, half=half
        ).model
    def __call__(self, img):
        # import time
        # t1 = time.time()
        h, w = img.shape[:2]
        features = self.model.get_features(np.array([(0,0,w,h)]),img)
        # print('reid time:', time.time() - t1, len(img_list))
        return features


def cosin_metric(x1, x2):   # (4, 512)*(512,)
    return np.dot(x1, x2) / (np.linalg.norm(x1, axis=1) * np.linalg.norm(x2))

if __name__ == '__main__':
    args = get_parser().parse_args()
    reid_model = Reid_feature(args.weights,args.device)
    query = []
    names = []
    # 检查文件是否存在
    query_file_path = os.path.join(args.query, 'query_features.npy')
    names_file_path = os.path.join(args.query, 'names.npy')

    if os.path.exists(query_file_path) and os.path.exists(names_file_path):
        query = np.load(query_file_path)
        names = np.load(names_file_path)
        print("Files loaded successfully.")
    else:
        print("One or both files do not exist.")
        query = np.ones((1, 512), dtype=np.int64)
        # print(q, q.shape)

        for person_name in os.listdir(args.query):
            if not os.path.isdir(os.path.join(args.query, person_name)):
                continue
            for image_name in os.listdir(os.path.join(args.query, person_name)):
                img = cv2.imread(os.path.join(args.query, person_name, image_name))
                feat = reid_model(img)   #normlized feat
                query = np.concatenate((feat, query), axis=0)
                names.append(person_name)
        names = names[::-1]
        names.append("None")
        # print(query[:-1, :].shape, names)
        np.save(os.path.join(args.query, 'query_features'), query[:-1, :])
        np.save(os.path.join(args.query, 'names'), names)  # save query
    # print(query.shape)
    # t1 = time.time()
    for image_name in os.listdir(os.path.join(args.input)):
        img = cv2.imread(os.path.join(args.input, image_name))
        t1 = time.time()
        feat = reid_model(img)   #normlized feat
        print('pytorch time:', time.time() - t1)
        cos_sim = cosine_similarity(feat, query)
        print(cos_sim)
        max_idx = np.argmax(cos_sim, axis=1)
        maximum = np.max(cos_sim, axis=1)
        max_idx[maximum < 0.6] = -1
        score = maximum
        results = max_idx
        print(score, results)
        label = names[results]
        print(label)
    # sim = cosin_metric(embs, q)
