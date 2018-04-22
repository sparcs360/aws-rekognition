#!/usr/bin/python3
import cv2
import numpy as np
import boto3

def create_rekognition_client(region = 'eu-west-1'):

    return boto3.client('rekognition', region)


def detect_labels(client, bucket='lnewfeld', filename=None, image_data=None):

    if not ((filename is None) ^ (image_data is None)):
        raise ValueError("filename xor image_data must be provided")
    
    if (image_data is None):
        image = {'S3Object': {'Bucket':bucket, 'Name':filename}}
    else:
        image = {'Bytes': image_data}

    #print("{}".format(image))
    return client.detect_labels(Image = image)


def get_snapshot(video_capture):

    while True:

        ret, frame = video_capture.read()
        #print("ret={}".format(ret))
        cv2.imshow('Video', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            ret, jpeg = cv2.imencode('.jpg', frame)
            #print("ret={}, jpeg={}".format(ret, jpeg))
            image_str = np.array(jpeg).tostring()
            return image_str
        if key == ord('q'):
            print("Aborting")
            return None

if __name__ == "__main__":

    video_capture = cv2.VideoCapture(0)

    FONT = cv2.FONT_HERSHEY_DUPLEX
    WIDTH = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    print("Creating client...")
    client = create_rekognition_client()

    while True:

        snapshot = get_snapshot(video_capture)
        if snapshot is None:
            break

        print("Detecting Labels from snapshot...")
        response = detect_labels(client, image_data=snapshot)

        print('Detected labels...')    
        for label in response['Labels']:
            print (label['Name'] + ' : ' + str(label['Confidence']))

    video_capture.release()
    cv2.destroyAllWindows()
