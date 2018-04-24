#!/usr/bin/python3
import cv2
import numpy as np
import boto3
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


def parse_command_line():
    import re

    def valid_face_name(value, regex=re.compile(r"^[a-zA-Z0-9_.\-:]+$")):
        if not regex.match(value):
            raise argparse.ArgumentTypeError("value must match regex {}".format(regex.pattern))
        return value

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", help="AWS region to use", default="eu-west-1")
    parser.add_argument("--collection-id", help="AWS rekognition collection to store known faces in",
                        default="lnewfeld")
    parser.add_argument("--face-name", type=valid_face_name, help="Who's in front of the camera?", default="NOONE")
    args = parser.parse_args()
    print("{}".format(args))
    return args


def create_rekognition_client(region):
    return boto3.client('rekognition', region)


def video_frame_to_jpeg_string(frame):
    ret, jpeg = cv2.imencode('.jpg', frame)
    image_str = np.array(jpeg).tostring()
    return image_str


def detect_objects(frame):
    request = {'Bytes': video_frame_to_jpeg_string(frame)}
    response = client.detect_labels(Image=request)
    for object in response['Labels']:
        print("{} = {:.2f}%".format(object['Name'], object['Confidence']))


class Face:
    def __init__(self, image, box, caption="", landmarks=None):
        image_height, image_width, _ = image.shape

        self.top = int(max(0, box['Top'] * image_height))
        self.left = int(max(0, box['Left'] * image_width))
        width = int(box['Width'] * image_width)
        height = int(box['Height'] * image_height)
        self.right = self.left + width
        self.bottom = self.top + height
        print("({},{})-({},{})".format(self.top,self.left, self.right,self.bottom))
        self.image = image[self.top:self.bottom, self.left:self.right]

        self.caption = caption
        if landmarks is None:
            self.landmarks = None
        else:
            self.landmarks = [
                (int(landmark['X'] * image_width), int(landmark['Y'] * image_height)) for landmark in landmarks
            ]
        self.facebox_alpha = 1.0

    def recognise(self, collection_id):
        image = {'Bytes': video_frame_to_jpeg_string(self.image)}
        try:
            response = client.search_faces_by_image(
                CollectionId=collection_id,
                Image=image,
                MaxFaces=1
            )
        except ClientError as e:
            self.caption = "UNKNOWN"
            print("{}".format(e))
            return

        if len(response['FaceMatches']) == 1:
            face_match = response['FaceMatches'][0]
            face = face_match['Face']
            self.caption = "{} ({:.2f}%)".format(face['ExternalImageId'], face_match['Similarity'])
        else:
            self.caption = "UNKNOWN"

    def draw_overlay(self, image):
        overlay = image.copy()
        cv2.rectangle(overlay, (self.left, self.top), (self.right, self.bottom), (0, 0, 255), 2)
        cv2.rectangle(overlay, (self.left, self.bottom), (self.right, self.bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(overlay, self.caption, (self.left + 6, self.bottom - 6), FONT, 0.5, (255, 255, 255), 1)
        if self.landmarks is not None:
            for (x, y) in self.landmarks:
                cv2.circle(overlay, (x, y), 4, (255, 0, 0), cv2.FILLED)
        self.facebox_alpha = self.facebox_alpha - 0.05
        cv2.addWeighted(overlay, self.facebox_alpha, image, 1 - self.facebox_alpha, 0, image)


def detect_faces(frame, faces):
    image = {'Bytes': video_frame_to_jpeg_string(frame)}
    start = time.time()
    response = client.detect_faces(Image=image)
    print("detect_faces: {}".format(time.time() - start))
    for face_detail in response['FaceDetails']:
        faces.append(Face(frame, box=face_detail['BoundingBox'], landmarks=face_detail['Landmarks']))


def index_face(image, collection_id, face_name):
    if face_name == "NOONE":
        print("Can't index unless you specify --face-name")
        return

    image = {'Bytes': video_frame_to_jpeg_string(image)}

    response = client.detect_faces(Image=image)
    if len(response['FaceDetails']) != 1:
        print("Indexing only allowed when one face detected (found {})".format(len(response['FaceDetails'])))
        return

    response = client.index_faces(
        CollectionId=collection_id,
        Image=image,
        ExternalImageId=face_name,
        DetectionAttributes=['ALL']
    )
    print("{}".format(response))


def recognise_faces(frame, collection_id, faces):
    image = {'Bytes': video_frame_to_jpeg_string(frame)}
    start = time.time()
    response = client.detect_faces(Image=image)
    print("detect_faces: {}".format(time.time() - start))
    details = response['FaceDetails']

    def recognise(box):
        face = Face(frame, box=box)
        start = time.time()
        face.recognise(collection_id)
        print("- recognise: {}".format(time.time() - start))
        return face

    start = time.time()
    futures = {executor.submit(recognise, detail['BoundingBox']): detail for detail in details}
    for future in as_completed(futures):
        face = future.result()
        faces.append(face)
        cv2.imshow(face.caption, face.image)
    print("recognise_total: {}".format(time.time() - start))


if __name__ == "__main__":
    args = parse_command_line()

    #boto3.set_stream_logger(name='botocore')

    client = create_rekognition_client(args.region)
    video_capture = cv2.VideoCapture(0)
    FONT = cv2.QT_FONT_NORMAL

    faces = []

    executor = ThreadPoolExecutor()

    while True:
        ret, frame = video_capture.read()

        key = cv2.waitKey(1) & 0xFF

        if key == ord('o'):
            print('Detecting objects...')
            detect_objects(frame)

        if key == ord('f'):
            print('Detecting faces...')
            detect_faces(frame, faces)

        if key == ord('i'):
            print('Indexing face...')
            index_face(frame, args.collection_id, args.face_name)

        if key == ord('r'):
            print('Looking for known faces...')
            recognise_faces(frame, args.collection_id, faces)

        if key == ord('t'):
            print('Looking for known faces in croud.jpg...')
            static_image = cv2.imread("croud.jpg")
            cv2.imshow("Croud...", static_image)
            croud_faces = []
            #detect_faces(static_image, croud_faces)
            recognise_faces(static_image, args.collection_id, croud_faces)
            for lee in croud_faces:
                lee.draw_overlay(static_image)
            cv2.imshow("Croud...", static_image)

        if key == ord('q'):
            print("Quitting")
            break

        if faces:
            for face in faces:
                face.draw_overlay(frame)
            faces = [face for face in faces if face.facebox_alpha > 0]

        cv2.imshow('Video', frame)

    video_capture.release()
    cv2.destroyAllWindows()
