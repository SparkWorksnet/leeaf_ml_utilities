import json
import os
from datetime import datetime

import tensorflow as tf
from flask import Flask, request
from werkzeug.exceptions import HTTPException

from image_utils import image_coordinates
from twinbase_utils import find_closest_tree, load_twinbase_tree_pois
from yolo_utils import run_predictions_on_image, load_model

pois_filename = 'trees.json'
segmentation_model_filename = '../bench_models/model-m.pt'
classification_model_filename = '../bench_models/vgg19'

tree_pois = None
if os.path.exists(pois_filename):
    print(f'loading pois...')
    f = open(pois_filename)
    tree_pois = json.load(f)['trees']
    print(f'loaded pois')
else:
    tree_pois = load_twinbase_tree_pois()

# load segmentation model
print(f'loading segmentation model...')
segmentation_model_filename = "../bench_models/model-m.pt"
model = load_model(segmentation_model_filename)
print(f'loaded: {segmentation_model_filename}')

# load classification model
print(f'loading classification model...')
classification_model = tf.keras.models.load_model(classification_model_filename)
print(f'loaded: {classification_model_filename}')

# start web server
print(f'loading web server...')
leeafEdge = Flask(__name__)


@leeafEdge.route("/")
def get_home():
    return {'service': 'up',
            'segmentation_model': segmentation_model_filename,
            'classification_model': classification_model_filename}


@leeafEdge.route("/pois")
def get_trees():
    return {'pois': tree_pois}


@leeafEdge.route('/<group>/<thing>/upload', methods=['POST'])
def uploadWithArgs(group, thing):
    return parse_uploaded_image(group=group, thing=thing)


@leeafEdge.route('/upload', methods=['POST'])
def upload():
    return parse_uploaded_image()


def parse_uploaded_image(group=None, thing=None):
    if len(request.files.getlist('file')) > 0:
        try:
            filename = download_file(request.files.getlist('file')[0])
            img_coordinates = None
            try:
                img_coordinates = image_coordinates(filename)
            except KeyError as k:
                pass
            (tree, distance) = find_closest_tree(tree_pois, img_coordinates)
            print(tree)
            if group is None:
                group = tree['group']
            if thing is None:
                tree_name = tree['uuid']
            else:
                tree_name = thing
            date_string = datetime.now().strftime("%Y%m%d")
            print(f'uploaded image for group:{group}, tree:{tree_name}, date: {date_string}, filename: {filename}')
            count, predict_time, detections = run_predictions_on_image(model, classification_model, image_path="./",
                                                                       image_name=filename, group=group, tree=tree_name,
                                                                       date_string=date_string)
            print(detections)
            return {'tree': tree['name'], 'distance': distance, 'count': count, 'predict_time': predict_time}
        except Exception as err:
            raise err
    else:
        raise NoFilesProvided()


@leeafEdge.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    # start with the correct headers and status code from the error
    response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,
    })
    response.content_type = "application/json"
    return response


class NoFilesProvided(HTTPException):
    code = 400
    description = 'no files found in request.'


def download_file(file):
    file.save(file.filename)
    return file.filename


if __name__ == "__main__":
    # leeafEdge.config()
    leeafEdge.run(host="0.0.0.0", port=30000, debug=False)
