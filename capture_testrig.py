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


if __name__ == "__main__":
    client = create_rekognition_client()
    video_capture = cv2.VideoCapture(0)
    FONT = cv2.FONT_HERSHEY_DUPLEX
    WIDTH = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    facebox_alpha = 0.0

    while True:

        ret, frame = video_capture.read()
        # print("ret={}".format(ret))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('o'):
            print('Detecting objects...')
            request = create_rekognition_request(image_data=video_frame_to_jpeg_string(frame))
            response = client.detect_labels(Image=request)
            for object in response['Labels']:
                print("{} = {:.2f}%".format(object['Name'], object['Confidence']))

        if key == ord('f'):
            print('Detecting faces...')
            request = create_rekognition_request(image_data=video_frame_to_jpeg_string(frame))
            response = client.detect_faces(Image=request)
            for face_detail in response['FaceDetails']:
                bounding_box = face_detail['BoundingBox']
                print("bounding_box={}".format(bounding_box))
                top = int(bounding_box['Top'] * HEIGHT)
                left = int(bounding_box['Left'] * WIDTH)
                width = int(bounding_box['Width'] * WIDTH)
                height = int(bounding_box['Height'] * HEIGHT)
                right = left + width
                bottom = top + height
                facebox_alpha = 1.0
                print("tl=({},{}), br=({},{})".format(top, left, right, bottom))

        if key == ord('q'):
            print("Quitting")
            break

        if facebox_alpha > 0:
            overlay = frame.copy()
            cv2.rectangle(overlay, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(overlay, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(overlay, "face", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            cv2.addWeighted(overlay, facebox_alpha, frame, 1 - facebox_alpha, 0, frame)
            facebox_alpha = facebox_alpha - 0.01

        cv2.imshow('Video', frame)

    video_capture.release()
    cv2.destroyAllWindows()
