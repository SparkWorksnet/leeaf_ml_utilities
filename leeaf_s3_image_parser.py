import io
import sys
from datetime import datetime

import boto3
from PIL import Image
from ultralytics import YOLO


def handle_yolo(model, image_path, bucket=None):
    print(f'[handle_yolo] bucket:{bucket}, image_path:{image_path}')

    s3_client = boto3.client('s3')
    image_name = image_path.split('/')[-1]
    s3_client.download_file('leeaf-datasets', image_path, image_name)

    image = Image.open(image_name)
    detections = model.predict(source=image_name, conf=CONF, iou=IOU, imgsz=SIZE)
    for detection_index, detection in enumerate(detections):
        label = detection.names[detection_index]
        for idx, item in enumerate(detection.boxes.xyxy.cpu().numpy()):
            confidence = int(detection.boxes.conf[detection_index].numpy() * 100)
            print(f'name={image_name} label={label} confidence={confidence}')
            box = (item[0], item[1], item[2], item[3])
            print(box)
            c_image = image.crop(box)
            # c_image.save(f'{image_name}_{idx}_{label}_{confidence}.jpg')

            tmp_image_bytes = io.BytesIO()
            c_image.save(tmp_image_bytes, format='JPEG')
            tmp_image_bytes.seek(0)

            dest_image_name = f"{image_name}_{idx:02}_{label}_{confidence:03}.jpg"
            datename = datetime.now().strftime('%Y%m%d')
            tree_image_name = f"public/{image_name.split('_')[0]}/{image_name.split('_')[1]}/{datename}/{dest_image_name}"
            print(tree_image_name)
            s3_client.put_object(Body=tmp_image_bytes, Bucket=bucket, Key=tree_image_name)  # c_image_name


model_file = sys.argv[1]
bucket_name = sys.argv[2]
bucket_path = sys.argv[3]

SIZE, CONF, IOU = 640, 0.2, 0.45
model = YOLO(model_file)

s3_client = boto3.client('s3')
items = s3_client.list_objects(Bucket=bucket_name, Prefix=bucket_path)
for item in items['Contents']:
    if '.jpg' in item['Key']:
        print(item)
        handle_yolo(model, image_path=item['Key'], bucket=bucket_name)
