import face_recognition
from PIL import Image, ImageDraw
import numpy as np
import os
import cv2
from ultralytics import YOLO
class FaceDetector:
    def __init__(self, host_image_path):
        # Load a sample picture and learn how to recognize it.
        host_image = face_recognition.load_image_file(host_image_path)
        face_locations = face_recognition.face_locations(host_image)
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
        face_locations = face_recognition.face_locations(unknown_image)
        face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
        face_ids = []
        face_locations_x1y1x2y2 = []
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
        return face_ids,face_locations_x1y1x2y2,face_encodings
    
class FaceDetectorV2:
    def __init__(self, host_image_path):
        # Load a sample picture and learn how to recognize it.
        host_image = face_recognition.load_image_file(host_image_path)
        face_locations = face_recognition.face_locations(host_image)
        host_face_encoding = face_recognition.face_encodings(host_image,face_locations)[0]

        # Create arrays of known face encodings and their names
        self.known_face_encodings = [
            host_face_encoding
        ]
        self.known_face_names = [
            "zk"
        ]
        self.yolo_model = YOLO("tracking/weights/yolov11m_best.pt")  # build from YAML and transfer weights

    def detect_faces(self, unknown_image):
        # Find all the faces and face encodings in the unknown image
        boxes = self.yolo_model.predict(unknown_image)[0].boxes.data
        face_locations = []
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box.cpu().numpy()
            if cls == 1:
                face_locations.append((y1,x2,y2,x1))
        face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
        face_ids = []
        face_locations_x1y1x2y2 = []
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
        return face_ids,face_locations_x1y1x2y2,face_encodings

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
    # This is an example of running face recognition on a single image
    # and drawing a box around each person that was identified.
    face_detector = FaceDetector("/home/jiangziben/data/people_tracking/known_people/zk/zk6.jpeg")

    # Load an image with an unknown face
    folder_path = "/home/jiangziben/data/people_tracking/zk"
    person_name = folder_path.split("/")[-1]
    images_path = get_images_in_folder(folder_path)
    num_all = len(images_path)
    num_right = 0
    for image_path in images_path:
        unknown_image = face_recognition.load_image_file(image_path)
        face_ids,face_locations,face_encodings = face_detector.detect_faces(unknown_image)
        # Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
        # See http://pillow.readthedocs.io/ for more about PIL/Pillow
        pil_image = Image.fromarray(unknown_image)
        # Create a Pillow ImageDraw Draw instance to draw with
        draw = ImageDraw.Draw(pil_image)

        # Loop through each face found in the unknown image
        for face_id, (left, top, right, bottom), face_encoding in zip(face_ids, face_locations, face_encodings):
            if face_id == -1:
                name = "Unknown"
            else:
                name = face_detector.known_face_names[face_id]
            if(person_name == name):
                num_right += 1
            # Draw a box around the face using the Pillow module
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

            # # Draw a label with a name below the face
            text_width, text_height = 50,50
            # draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
            draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

        # Remove the drawing library from memory as per the Pillow docs
        del draw
        # Display the resulting image
        # pil_image.show()
        cv2.imshow('image',np.array(pil_image))
        cv2.waitKey(0)
    accuracy = num_right / num_all 
    print(f"accuracy: {accuracy*100.0:.2f} %")
    # You can also save a copy of the new image to disk if you want by uncommenting this line
    # pil_image.save("image_with_boxes.jpg")
