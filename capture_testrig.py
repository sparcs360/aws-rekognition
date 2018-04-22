#!/usr/bin/python3
import cv2
import numpy as np
import boto3
from botocore.exceptions import ClientError


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
    def __init__(self, box, caption="", landmarks=None):
        self.top = int(box['Top'] * HEIGHT)
        self.left = int(box['Left'] * WIDTH)
        width = int(box['Width'] * WIDTH)
        height = int(box['Height'] * HEIGHT)
        self.right = self.left + width
        self.bottom = self.top + height
        self.caption = caption
        if landmarks is None:
            self.landmarks = None
        else:
            self.landmarks = [
                (int(landmark['X'] * WIDTH), int(landmark['Y'] * HEIGHT)) for landmark in landmarks
            ]
        self.facebox_alpha = 1.0

    def draw_overlay(self, frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (self.left, self.top), (self.right, self.bottom), (0, 0, 255), 2)
        cv2.rectangle(overlay, (self.left, self.bottom), (self.right, self.bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(overlay, self.caption, (self.left + 6, self.bottom - 6), FONT, 0.5, (255, 255, 255), 1)
        if self.landmarks is not None:
            for (x, y) in self.landmarks:
                cv2.circle(overlay, (x, y), 4, (255, 0, 0), cv2.FILLED)
        self.facebox_alpha = self.facebox_alpha - 0.1
        cv2.addWeighted(overlay, self.facebox_alpha, frame, 1 - self.facebox_alpha, 0, frame)


def detect_faces(frame, faces):
    image = {'Bytes': video_frame_to_jpeg_string(frame)}
    response = client.detect_faces(Image=image)
    for face_detail in response['FaceDetails']:
        faces.append(Face(box=face_detail['BoundingBox'], landmarks=face_detail['Landmarks']))


def index_face(frame, collection_id, face_name):
    if face_name == "NOONE":
        print("Can't index unless you specify --face-name")
        return

    image = {'Bytes': video_frame_to_jpeg_string(frame)}

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

    try:
        response = client.search_faces_by_image(
            CollectionId=collection_id,
            Image=image,
            MaxFaces=2
        )
        print("{}".format(response))
    except ClientError as e:
        print("{}".format(e))
        return

    for face_match in response['FaceMatches']:
        face = face_match['Face']
        caption = "{} ({:.2f}%)".format(face['ExternalImageId'], face_match['Similarity'])
        faces.append(Face(caption=caption, box=response['SearchedFaceBoundingBox']))


if __name__ == "__main__":
    args = parse_command_line()

    client = create_rekognition_client(args.region)
    video_capture = cv2.VideoCapture(0)
    FONT = cv2.QT_FONT_NORMAL
    WIDTH = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    faces = []

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
