import face_recognition
from PIL import Image, ImageDraw
import numpy as np
import os
import cv2
from ultralytics import YOLO
from tracking.get_reid_feature import Reid_feature
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from pathlib import PosixPath
from tracking.face_location import get_face_location_from_keypoints,draw_face_bboxes
import torch
import time

class FaceDetector:
    def __init__(self, host_image_path):
        # Load a sample picture and learn how to recognize it.
        host_image = face_recognition.load_image_file(host_image_path)
        face_locations = face_recognition.face_locations(host_image,model = "cnn")
        host_face_encoding = face_recognition.face_encodings(host_image,face_locations)[0]
        host_name = host_image_path.split("/")[-2]
        # Create arrays of known face encodings and their names
        self.known_face_encodings = [
            host_face_encoding
        ]
        self.known_face_names = [
            host_name
        ]
    def detect_faces(self, unknown_image):
        # Find all the faces and face encodings in the unknown image
        face_locations = face_recognition.face_locations(unknown_image,model="cnn")
        face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
        face_ids = []
        face_locations_x1y1x2y2 = []
        face_confidences = []
        # Loop through each face found in the unknown image
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding,tolerance=0.4)

            face_id = -1

            # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                face_id = best_match_index
            face_ids.append(face_id)
            face_locations_x1y1x2y2.append((left,top,right,bottom))
            face_confidences.append(1-face_distances[best_match_index])
        return face_ids,face_locations_x1y1x2y2,face_encodings,face_confidences
    
class FaceDetectorV2:
    def __init__(self,detection_model,reid_model, host_image_path):
        if isinstance(detection_model, str) or isinstance(detection_model, PosixPath):
            self.detection_model = YOLO(detection_model)  # build from YAML and transfer weights
        else:
            self.detection_model = detection_model
        if isinstance(reid_model, str) or isinstance(detection_model, PosixPath):
            self.reid_model = Reid_feature(reid_model)
        else:
            self.reid_model = reid_model
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        # 检查文件是否存在
        query_file_path = os.path.join(host_image_path, 'query_features.npy')
        names_file_path = os.path.join(host_image_path, 'names.npy')
        ids_file_path = os.path.join(host_image_path, 'ids.npy')
        if os.path.exists(query_file_path) and os.path.exists(names_file_path) and os.path.exists(ids_file_path):
            self.known_face_encodings = np.load(query_file_path)
            self.known_face_ids = np.load(ids_file_path)
            self.known_face_names = np.load(names_file_path)
            print("Files loaded successfully.")
        else:
            print("One or both files do not exist.")
            self.known_face_encodings = np.ones((1,128), dtype=np.int64)
            # print(q, q.shape)

            for i,person_name in enumerate(os.listdir(host_image_path)):
                if not os.path.isdir(os.path.join(host_image_path, person_name)):
                    continue
                for image_name in os.listdir(os.path.join(host_image_path, person_name)):
                    img = cv2.imread(os.path.join(host_image_path, person_name, image_name))
                    boxes = self.detection_model.predict(img)[0].boxes.data
                    face_locations = []
                    for box in boxes:
                        x1, y1, x2, y2, conf, cls = box.cpu().numpy()
                        if cls == 1:
                            face_locations.append((y1,x2,y2,x1))
                    face_encodings = face_recognition.face_encodings(img,face_locations)
                    # face_encodings = self.face_encodings(img,face_locations)
                    for face_encoding in face_encodings:
                        self.known_face_encodings = np.concatenate((face_encoding[np.newaxis,:], self.known_face_encodings), axis=0)
                    self.known_face_ids.append(i)
                self.known_face_names.append(person_name)
            self.known_face_ids = self.known_face_ids[::-1]
            self.known_face_ids.append(-1)
            np.save(os.path.join(host_image_path, 'query_features'), self.known_face_encodings[:-1, :])
            np.save(os.path.join(host_image_path, 'ids'), self.known_face_ids)  # save query
            np.save(os.path.join(host_image_path, 'names'), self.known_face_names)  # save query

    def face_encodings(self,img,face_locations):
        face_encodings = []
        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_image = img[int(top):int(bottom), int(left):int(right)]
            face_encoding = self.reid_model(face_image)
            face_encodings.append(face_encoding)
        return face_encodings

    def detect_faces(self, unknown_image, threshold = 0.4):
        # Find all the faces and face encodings in the unknown image
        boxes = self.detection_model.predict(unknown_image)[0].boxes.data
        face_locations = []
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box.cpu().numpy()
            if cls == 1:
                face_locations.append((y1,x2,y2,x1))
        face_encodings = face_recognition.face_encodings(unknown_image,face_locations)
        # face_encodings = self.face_encodings(unknown_image,face_locations)
        face_ids = []
        face_locations_x1y1x2y2 = []
        # Loop through each face found in the unknown image
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            # cos_sim = cosine_similarity(face_encoding[np.newaxis, :], self.known_face_encodings)
            # max_idx = np.argmax(cos_sim, axis=1)
            # maximum = np.max(cos_sim, axis=1)
            # max_idx[maximum < threshold] = -1
            # face_id = self.known_face_ids[max_idx[0]]
            euclidean_sim = euclidean_distances(face_encoding[np.newaxis, :], self.known_face_encodings)
            min_idx = np.argmin(euclidean_sim, axis=1)
            minimun = np.min(euclidean_sim, axis=1)
            min_idx[minimun > threshold] = -1
            face_id = self.known_face_ids[min_idx[0]]      
            face_ids.append(face_id)
            face_locations_x1y1x2y2.append((left,top,right,bottom))
        return face_ids,face_locations_x1y1x2y2,face_encodings

class FaceDetectorV3:
    def __init__(self,detection_model,reid_model, host_image_path):
        if isinstance(detection_model, str) or isinstance(detection_model, PosixPath):
            self.detection_model = YOLO(detection_model)  # build from YAML and transfer weights
        self.reid_model = reid_model
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        # 检查文件是否存在
        query_file_path = os.path.join(host_image_path, 'query_features.npy')
        names_file_path = os.path.join(host_image_path, 'names.npy')
        ids_file_path = os.path.join(host_image_path, 'ids.npy')
        if os.path.exists(query_file_path) and os.path.exists(names_file_path) and os.path.exists(ids_file_path):
            self.known_face_encodings = np.load(query_file_path)
            self.known_face_ids = np.load(ids_file_path)
            self.known_face_names = np.load(names_file_path)
            print("Files loaded successfully.")
        else:
            print("One or both files do not exist.")
            self.known_face_encodings = np.ones((1,128), dtype=np.int64)
            # print(q, q.shape)
            for i,person_name in enumerate(os.listdir(host_image_path)):
                if not os.path.isdir(os.path.join(host_image_path, person_name)):
                    continue
                for image_name in os.listdir(os.path.join(host_image_path, person_name)):
                    img = face_recognition.load_image_file(os.path.join(host_image_path, person_name, image_name))
                    people_keypoints = self.detection_model.predict(img)[0].keypoints.data
                    face_locations = []
                    for person_keypoints in people_keypoints:
                        face_bbox = get_face_location_from_keypoints(person_keypoints,img.shape)                        
                        if face_bbox is not None:
                            (left, top, right, bottom) = map(int,face_bbox)
                            face_locations.append([top, right, bottom, left])
                    face_encodings = face_recognition.face_encodings(img,face_locations)
                    for face_encoding in face_encodings:
                        if len(self.known_face_encodings) == 0:
                            self.known_face_encodings = face_encoding[np.newaxis,:]
                        else:
                            self.known_face_encodings = np.concatenate((face_encoding[np.newaxis,:], self.known_face_encodings), axis=0)
                    self.known_face_ids.append(i)
                self.known_face_names.append(person_name)
            self.known_face_ids = self.known_face_ids[::-1]
            self.known_face_ids.append(-1)
            np.save(os.path.join(host_image_path, 'query_features'), self.known_face_encodings)
            np.save(os.path.join(host_image_path, 'ids'), self.known_face_ids)  # save query
            np.save(os.path.join(host_image_path, 'names'), self.known_face_names)  # save query

    def face_encodings(self,img,face_locations):
        face_encodings = []
        for face_location in face_locations:
            left, top, right, bottom = face_location
            face_image = img[int(top):int(bottom), int(left):int(right)]
            face_image = cv2.resize(face_image, (640, 640))
            face_image_tensor = torch.from_numpy(face_image).float()
            face_image_tensor = face_image_tensor.permute(2, 0, 1)
            face_encoding = self.reid_model(face_image_tensor.unsqueeze(0))
            face_encodings.append(face_encoding[0])
        return face_encodings
    
    def confidence_exponential(self,d, alpha=1.0):
        return np.exp(-alpha * d)
    
    def detect_faces(self, unknown_image, threshold = 0.6):
    
        # Find all the faces and face encodings in the unknown image
        people_keypoints = self.detection_model.predict(unknown_image,verbose=False)[0].keypoints.data
        person_indexes = []
        face_locations = []
        for i,person_keypoints in enumerate(people_keypoints):
            face_bbox = get_face_location_from_keypoints(person_keypoints,unknown_image.shape)
            if face_bbox is not None:
                (left, top, right, bottom) = map(int,face_bbox)
                face_locations.append([top, right, bottom, left])
                person_indexes.append(i)
        face_encodings = face_recognition.face_encodings(unknown_image,face_locations)
        face_ids = []
        face_locations_x1y1x2y2 = []
        face_confidences = []
        # Loop through each face found in the unknown image
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            # cos_sim = cosine_similarity(face_encoding[np.newaxis, :], self.known_face_encodings)
            # max_idx = np.argmax(cos_sim, axis=1)
            # maximum = np.max(cos_sim, axis=1)
            # max_idx[maximum < threshold] = -1
            # face_id = self.known_face_ids[max_idx[0]]
            euclidean_sim = euclidean_distances(face_encoding[np.newaxis, :], self.known_face_encodings)
            min_idx = np.argmin(euclidean_sim, axis=1)
            minimun = np.min(euclidean_sim, axis=1)
            face_confidence = self.confidence_exponential(minimun)
            min_idx[face_confidence < threshold] = -1
            if min_idx[0] >= 0:
                face_id = self.known_face_ids[min_idx[0]] 
            else:
                face_id = -1
            face_ids.append(face_id)
            face_locations_x1y1x2y2.append([left, top, right, bottom])
            face_confidences.append(face_confidence[0])
        return face_ids,face_locations_x1y1x2y2,face_encodings,face_confidences
    
    def detect_faces(self, unknown_image,detect_result, threshold = 0.7):
        people_keypoints = detect_result.keypoints.data
        # Find all the faces and face encodings in the unknown image
        person_indexes = []
        face_locations = []
        for i,person_keypoints in enumerate(people_keypoints):
            face_bbox = get_face_location_from_keypoints(person_keypoints,unknown_image.shape)
            if face_bbox is not None:
                (left, top, right, bottom) = map(int,face_bbox)
                face_locations.append([top, right, bottom, left])
                person_indexes.append(i)
        face_encodings = face_recognition.face_encodings(unknown_image,face_locations)
        face_ids = []
        face_locations_x1y1x2y2 = []
        face_confidences = []
        # Loop through each face found in the unknown image
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            # cos_sim = cosine_similarity(face_encoding[np.newaxis, :], self.known_face_encodings)
            # max_idx = np.argmax(cos_sim, axis=1)
            # maximum = np.max(cos_sim, axis=1)
            # max_idx[maximum < threshold] = -1
            # face_id = self.known_face_ids[max_idx[0]]
            euclidean_sim = euclidean_distances(face_encoding[np.newaxis, :], self.known_face_encodings)
            min_idx = np.argmin(euclidean_sim, axis=1)
            minimun = np.min(euclidean_sim, axis=1)
            face_confidence = self.confidence_exponential(minimun)
            min_idx[face_confidence < threshold] = -1
            if min_idx[0] >= 0:
                face_id = self.known_face_ids[min_idx[0]] 
            else:
                face_id = -1
            face_ids.append(face_id)
            face_locations_x1y1x2y2.append([left, top, right, bottom])
            face_confidences.append(face_confidence[0])
        return face_ids,face_locations_x1y1x2y2,face_encodings,face_confidences,person_indexes


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
    # 按照文件的修改时间进行排序
    images.sort(key=lambda x: os.path.getmtime(x))
    return images 
if __name__ == "__main__":
    # This is an example of running face recognition on a single image
    # and drawing a box around each person that was identified.
    face_detector = FaceDetectorV3("/home/jiangziben/CodeProject/boxmot/tracking/weights/yolov11m-pose.pt",
                                   "/home/jiangziben/CodeProject/boxmot/tracking/weights/20180402-114759-vggface2.pt",
                                   "/home/jiangziben/CodeProject/boxmot/data/known_people/")
    # face_detector = FaceDetector("/home/jiangziben/data/people_tracking/known_people/jzb/jzb.png")
    # Load an image with an unknown face
    folder_path = "/home/jiangziben/data/people_tracking/multi_people"
    person_name = folder_path.split("/")[-1]
    images_path = get_images_in_folder(folder_path)
    num_all = len(images_path)
    num_right = 0
    # while True:
    for image_path in images_path:
        unknown_image = face_recognition.load_image_file(image_path)
        start = time.time()
        detect_result = face_detector.detection_model.predict(unknown_image,verbose=False)[0]
        people_bboxes = detect_result.boxes.data
        face_ids,face_locations,face_encodings,face_confidences,person_indexes = face_detector.detect_faces(unknown_image,detect_result)
        print("used_times: ",time.time() - start)
        # Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
        # See http://pillow.readthedocs.io/ for more about PIL/Pillow
        pil_image = Image.fromarray(unknown_image)
        # Create a Pillow ImageDraw Draw instance to draw with
        draw = ImageDraw.Draw(pil_image)

        # Loop through each face found in the unknown image
        for face_id, face_location, face_encoding,face_confidence,person_index in zip(face_ids, face_locations, face_encodings,face_confidences,person_indexes):
            left, top, right, bottom = face_location
            person_box = people_bboxes[person_index]
            
            if face_id == -1:
                name = "Unknown"
            else:
                name = face_detector.known_face_names[face_id]
            if(person_name == name):
                num_right += 1
            
            
            # Draw a box around the face using the Pillow module
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
            draw.rectangle(((person_box[0], person_box[1]), (person_box[2], person_box[3])), outline=(0, 255, 0))

            # # Draw a label with a name below the face
            text_width, text_height = 50,50
            # draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
            draw.text((left + 6, bottom - text_height - 5), name + ", c: " + f"{face_confidence:.2f}", fill=(255, 0, 0, 255))

        # Remove the drawing library from memory as per the Pillow docs
        del draw
        # Display the resulting image
        # pil_image.show()
        cv2.imshow('image',np.array(pil_image)[:,:,::-1])
        cv2.waitKey(40)
    if num_all > 0:
        accuracy = num_right / num_all
    else:
        accuracy = 0
    print(f"accuracy: {accuracy*100.0:.2f} %")
    # You can also save a copy of the new image to disk if you want by uncommenting this line
    # pil_image.save("image_with_boxes.jpg")
