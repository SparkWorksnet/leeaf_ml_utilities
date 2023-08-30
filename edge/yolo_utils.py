import io
import pickle
import time
from datetime import datetime

import boto3
import keras.utils as kimage
import numpy as np
from PIL import Image
from ultralytics import YOLO

SIZE = 640
CONFIDENCE = 0.2
IOU_THRESHOLD = 0.45

labels = ['aculus_orealus', 'healthy', 'peacock_spot']


def run_predictions_on_image(seg_model, clas_model, image_path, image_name, group, tree, date_string):
    """
    Process a single image, upload predictions to cloud and return the detected results
    :param seg_model:
    :param clas_model:
    :param image_path:
    :param image_name:
    :param group:
    :param tree:
    :param date_string:
    :return: the detected instances of leaves
    """
    print(f'[handle_yolo] image_path:{image_name}')
    date_string = datetime.now().strftime('%Y%m%d')
    clean_image_name = image_name.split('.')[0]

    start_time = time.time()
    image = Image.open(f'{image_path}/{image_name}')
    detections = seg_model.predict(source=f'{image_path}/{image_name}', conf=CONFIDENCE, iou=IOU_THRESHOLD, imgsz=SIZE,
                                   save=True, save_crop=True,
                                   retina_masks=True)
    end_time = time.perf_counter()

    file = open('detections.p', 'wb')
    pickle.dump(detections, file)
    file.close()

    session = boto3.Session(profile_name='leeaf')
    s3_client = session.client('s3')
    tree_image_path = f"public/{group}/{tree}/{date_string}/{clean_image_name}"

    for detection_index, detection in enumerate(detections):
        label = detection.names[detection_index]
        for idx, item in enumerate(detection.boxes.xyxy.cpu().numpy()):
            confidence = int(detection.boxes.conf[detection_index].numpy() * 100)
            box = (item[0], item[1], item[2], item[3])
            print(f'name={image_name}, label={label}, confidence={confidence}, box={box}')
            c_image = image.crop(box)
            tmp_image_bytes = io.BytesIO()
            c_image.save(tmp_image_bytes, format='JPEG')
            tmp_image_bytes.seek(0)
            c_image.save('tmp.jpg')
            img = kimage.load_img(f'tmp.jpg', target_size=(256, 256))
            img = np.expand_dims(img, axis=0)
            result = clas_model.predict(img)
            new_label = labels[np.argmax(result)]

            dest_image_name = f'{new_label}_{idx:04}_{confidence:03}.jpg'
            s3_client.put_object(Body=tmp_image_bytes, Bucket='leeaf-datasets',
                                 Key=f'{tree_image_path}/{dest_image_name}')

    total_leaves = 0
    response = []
    for detection_index, detection in enumerate(detections):
        label = detection.names[detection_index]
        total_leaves = total_leaves + len(detection.boxes.xyxy.cpu().numpy())
        for item in detection.boxes.xyxy.cpu().numpy():
            print(item)
            response.append({'label': label})
            total_leaves += 1
    print(response)
    return total_leaves, (end_time - start_time), response


def load_model(model_file):
    """
    loads a YOLO model from file
    :param model_file: the file with the weights of the YOLO model
    :return: the loaded YOLO model
    """
    return YOLO(model_file)
