# aws-rekognition
Playing with AWS Rekognition (in Python3)

# Searching Faces in a Collection

https://docs.aws.amazon.com/rekognition/latest/dg/collections.html

## Create a collection of known faces

```bash
$ aws rekognition create-collection --collection-id lnewfeld
```

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

\(index a face by pressing `i` in `capture_testrig.py`)

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

```bash
$ aws rekognition delete-faces --collection-id lnewfeld --face-ids a132d053-6549-4b86-9332-fea5167e868c
{
    "DeletedFaces": [
        "a132d053-6549-4b86-9332-fea5167e868c"
    ]
}
```
