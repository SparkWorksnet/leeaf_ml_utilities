# LEEAF Project ML Utilities

Created with the support of the [IoT-Ngin 2nd Open Call](https://iot-ngin.eu/)

## Contents

This repository contains helpfull utility scripts and notebooks for the ML analysis of the data generated from the LEEAF
IoT-Ngin OC2 project.

### dataset_view

A helper notebook used to review and parse the datasets used in the project.

### leeaf_trainyolo_model_trainer

A jupyter notebook to train a leaf detection model using the trainYOLO platform.

### leeaf_s3_image_parser

An AWS lambda function to check newly collected images for leafs and generate cropped leaf images from them using a
YOLO-based ML model.

### Classification

[Classification](classification) contains a benchmark file and ML models that can be used to test the speed of the
classification process on different systems.

### Edge

[Edge](edge) contains the implementation of the edge server of LEEAF. An application that is designed to process image
files and upload the extracted leaves with their characterization to the cloud.
