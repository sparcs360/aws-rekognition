# aws-rekognition

Perform various operations on a frame of video input (from a webcam):
* Detect Objects
* Detect Faces
* Add a detected face to a collection of known faces (for later recognition)
* Recognise faces

## Requirements

* A video input (e.g., webcam)
* An AWS account
* Python3 (I'm using 3.6.5)
* PIP (https://packaging.python.org/tutorials/installing-packages/)
  * Boto3 (https://github.com/boto/boto3)
  * OpenCV (https://github.com/opencv/opencv)

# Build

## Install Python3

TODO

## Install PIP

TODO

## Virtual Environment

TODO

## Install OpenCV

TODO

## Install Boto3

```bash
pip install boto3
```

# Test

```bash
capture_testrig.py --help
```

Should show help on command-line arguments

# Configure

## AWS

You need to setup  the `aws` CLI (TODO: link)

TODO: More

### Create a collection to store known faces in

You must give the colection a unique Id (in the `collection-id` argument)

```bash
aws rekognition create-collection --collection-id lnewfeld
```

See the [Appendix](#aws-rekognition-collection-management) for more collection commands.

## Command-line

TODO

# User Guide

TODO

# Appendix

## AWS Rekognition Collection Management

### What collections have I created?

```bash
$ aws rekognition list-collections
{
    "FaceModelVersions": [
        "3.0"
    ],
    "CollectionIds": [
        "lnewfeld"
    ]
}
```

### What faces in the collection?

```bash
$ aws rekognition list-faces --collection-id lnewfeld
{
    "FaceModelVersion": "3.0",
    "Faces": [
        {
            "BoundingBox": {
                "Width": 0.29326900839805603,
                "Top": 0.31410300731658936,
                "Left": 0.4699519872665405,
                "Height": 0.39102599024772644
            },
            "FaceId": "a132d053-6549-4b86-9332-fea5167e868c",
            "ExternalImageId": "test",
            "Confidence": 99.99970245361328,
            "ImageId": "907c0e14-5268-52f2-9d78-1f38a2ad4cce"
        }
    ]
}
```

### Delete faces from a collection

```bash
$ aws rekognition delete-faces --collection-id lnewfeld --face-ids a132d053-6549-4b86-9332-fea5167e868c
{
    "DeletedFaces": [
        "a132d053-6549-4b86-9332-fea5167e868c"
    ]
}
```
