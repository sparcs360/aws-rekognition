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


if __name__ == "__main__":

    print("Creating client...")
    client = create_rekognition_client()

    print("Detecting Labels from an image in an S3 bucket...")
    response = detect_labels(client, filename='lee_fried_gold.jpg')

    print('Detected labels...')    
    for label in response['Labels']:
        print (label['Name'] + ' : ' + str(label['Confidence']))
