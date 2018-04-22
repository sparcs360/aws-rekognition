#!/usr/bin/python3
import cv2
import numpy as np
import boto3


def create_rekognition_client(region='eu-west-1'):
    return boto3.client('rekognition', region)


def create_rekognition_request(bucket='lnewfeld', filename=None, image_data=None):
    if not ((filename is None) ^ (image_data is None)):
        raise ValueError("filename xor image_data must be provided")

    if image_data is None:
        request = {'S3Object': {'Bucket': bucket, 'Name': filename}}
    else:
        request = {'Bytes': image_data}

    return request


def video_frame_to_jpeg_string(frame):
    ret, jpeg = cv2.imencode('.jpg', frame)
    image_str = np.array(jpeg).tostring()
    return image_str


def detect_objects(frame):
    print('Detecting objects...')
    request = create_rekognition_request(image_data=video_frame_to_jpeg_string(frame))
    response = client.detect_labels(Image=request)
    for object in response['Labels']:
        print("{} = {:.2f}%".format(object['Name'], object['Confidence']))


class Face:
    def __init__(self, face_detail):
        bounding_box = face_detail['BoundingBox']
        self.top = int(bounding_box['Top'] * HEIGHT)
        self.left = int(bounding_box['Left'] * WIDTH)
        width = int(bounding_box['Width'] * WIDTH)
        height = int(bounding_box['Height'] * HEIGHT)
        self.right = self.left + width
        self.bottom = self.top + height
        self.facebox_alpha = 1.0


def detect_faces(frame, faces):
    print('Detecting faces...')
    request = create_rekognition_request(image_data=video_frame_to_jpeg_string(frame))
    response = client.detect_faces(Image=request)
    for face_detail in response['FaceDetails']:
        faces.append(Face(face_detail))


if __name__ == "__main__":
    client = create_rekognition_client()
    video_capture = cv2.VideoCapture(0)
    FONT = cv2.FONT_HERSHEY_DUPLEX
    WIDTH = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    faces = []

    while True:
        ret, frame = video_capture.read()

        key = cv2.waitKey(1) & 0xFF

        if key == ord('o'):
            detect_objects(frame)

        if key == ord('f'):
            detect_faces(frame, faces)

        if key == ord('q'):
            print("Quitting")
            break

        print("face_count={}".format(len(faces)))
        if faces:
            for face in faces:
                overlay = frame.copy()
                cv2.rectangle(overlay, (face.left, face.top), (face.right, face.bottom), (0, 0, 255), 2)
                cv2.rectangle(overlay, (face.left, face.bottom), (face.right, face.bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(overlay, "face", (face.left + 6, face.bottom - 6), FONT, 1.0, (255, 255, 255), 1)
                face.facebox_alpha = face.facebox_alpha - 0.01
                cv2.addWeighted(overlay, face.facebox_alpha, frame, 1 - face.facebox_alpha, 0, frame)
            faces = [face for face in faces if face.facebox_alpha > 0]

        cv2.imshow('Video', frame)

    video_capture.release()
    cv2.destroyAllWindows()
