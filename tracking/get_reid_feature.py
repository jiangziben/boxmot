
import argparse
from pathlib import Path
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reid-model', type=Path, default="/home/jiangziben/CodeProject/boxmot/tracking/weights/osnet_x0_25_msmt17.pt",
                        help='reid model path')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')


    opt = parser.parse_args()




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

def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument('--reid-model', type=Path, default="/home/jiangziben/CodeProject/boxmot/tracking/weights/osnet_x0_25_msmt17.pt",
                        help='reid model path')
    parser.add_argument(
        "--input",
        # nargs="+",
        type=Path,
        default='/home/jiangziben/CodeProject/boxmot/datasets/person_bank',
        help="path to known peopel image folders",
    )
    parser.add_argument(
        "--output",
        default='demo_output',
        help='path to save features'
    )
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')

    return parser


class Reid_feature():
    def __init__(self,args):
        self.model = ReidAutoBackend(
        weights=args.reid_model, device=args.device, half=True
        ).model
    def __call__(self, img_list):
        import time
        t1 = time.time()
        h, w = img.shape[:2]
        features = self.model.get_features(np.array([(0,0,w,h)]),img_list)
        # print('reid time:', time.time() - t1, len(img_list))
        return features


def cosin_metric(x1, x2):   # (4, 512)*(512,)
    return np.dot(x1, x2) / (np.linalg.norm(x1, axis=1) * np.linalg.norm(x2))


if __name__ == '__main__':
    args = get_parser().parse_args()
    reid_model = Reid_feature(args)
    embs = []
    names = []
    embs = np.ones((1, 512), dtype=np.int64)
    # print(q, q.shape)
    for person_name in os.listdir(args.input):
        if not os.path.isdir(os.path.join(args.input, person_name)):
            continue
        for image_name in os.listdir(os.path.join(args.input, person_name)):
            img = cv2.imread(os.path.join(args.input, person_name, image_name))

            t1 = time.time()
            feat = reid_model(img)   #normlized feat
            print('pytorch time:', time.time() - t1)
            embs = np.concatenate((feat, embs), axis=0)
            names.append(person_name)
            # print(embs.shape, names)
            # print('====sim:', sim)
    names = names[::-1]
    names.append("None")
    print(embs[:-1, :].shape, names)

    # print(query.shape)
    # np.save(os.path.join(args.input, 'query_features'), embs[:-1, :])
    # np.save(os.path.join(args.input, 'names'), names)  # save query
    # t1 = time.time()
    query = np.load(os.path.join(args.input,'query_features.npy'))
    cos_sim = cosine_similarity(embs, query)
    print(cos_sim)
    max_idx = np.argmax(cos_sim, axis=1)
    maximum = np.max(cos_sim, axis=1)
    max_idx[maximum < 0.6] = -1
    score = maximum
    results = max_idx
    print(score, results)
    for i in range(5):
        label = names[results[i]]
        print(label)
    # sim = cosin_metric(embs, q)



        # np.testing.assert_allclose(pytorch_output, onnx_output, rtol=1e-4, atol=1e-6)
        # print("恭喜你 ^^ ，onnx 和 pytorch 结果一致 ， Exported model has been executed decimal=5 and the result looks good!")

        # np.save(os.path.join(args.output, path.replace('.jpg', '.npy').split('/')[-1]), feat)

