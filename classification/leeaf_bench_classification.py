import os
import time

import tensorflow as tf

test_image_path = "dataset/test"

IMAGE_SIZE = [256, 256]

test_dataset = tf.keras.utils.image_dataset_from_directory(directory=test_image_path, image_size=IMAGE_SIZE,
                                                           batch_size=1, shuffle=False)

models_dir = "../bench_models"
prediction_times = {}
dataset_size = len(test_dataset)

for model in os.listdir(models_dir):
    if model.endswith('.h5'):
        print(model)
        existing_model = tf.keras.models.load_model(model)
        start = time.time()
        y_pred = existing_model.predict(test_dataset)
        time_diff = time.time() - start
        prediction_times[model] = time_diff / dataset_size
print(prediction_times)
