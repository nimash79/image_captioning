# Copyright (c) 2026 Nima Sharifinia
# Licensed under the Apache License, Version 2.0

path = "/home/nima/Downloads/"
features_type = "xception"

import json
import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.utils.data import Dataset
from PIL import Image
import os
import sys
from torchvision import transforms
import re
from collections import Counter
import matplotlib.pyplot as plt
from collections import defaultdict
import csv

class CocoDataSet(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.all_captions = []
        self.all_images = []
        self.all_ids = []
        self.max_length = 0
        self.PATH = f"{path}/coco2017/train2017/"

        # Load annotations
        with open(f'{path}/coco2017/annotations/captions_train2017.json', 'r') as file:
            data = json.load(file)

        # Create an image index
        image_id_index = {}
        for img in data['images']:
            image_id_index[img['id']] = img['file_name']

        for annot in data['annotations']:
            caption = '<start> ' + annot['caption'] + ' <end>'
            image_id = annot['image_id']
            full_coco_image_path = self.PATH + image_id_index[image_id]
            if not image_id in self.all_ids:
                self.all_ids.append(image_id)
                self.all_images.append(full_coco_image_path)
                self.all_captions.append(caption)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.all_captions)

    def __getitem__(self, idx):
        """Fetches the image and encoded caption at the specified index."""
        image_name = self.all_images[idx]
        return image_name

# Example usage
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.Lambda(lambda img: img.convert("RGB")),  # Ensure 3 channels (convert grayscale to RGB)
    transforms.ToTensor(),
])

# Create the custom dataset
dataset = CocoDataSet(transform=transform)
dataset_len = len(dataset)

def partition(arr, low, high): 
	i = (low - 1)  # index of smaller element 
	pivot = arr[high]  # pivot 

	for j in range(low, high): 

		# If current element is smaller than or 
		# equal to pivot 
		if arr[j][6] >= pivot[6]:  # the width of the box times the height of the box times the confidence rate # hadie

			# increment index of smaller element 
			i = i + 1
			arr[i], arr[j] = arr[j], arr[i] 

	arr[i + 1], arr[high] = arr[high], arr[i + 1] 
	return (i + 1) 

def quickSort(arr, low, high): 
	if len(arr) == 1: 
		return arr 
	if low < high: 

		# pi is partitioning index, arr[p] is now 
		# at right place 
		pi = partition(arr, low, high) 

		# Separately sort elements before 
		# partition and after partition 
		quickSort(arr, low, pi - 1) 
		quickSort(arr, pi + 1, high) 
            
import tensorflow as tf
from tensorflow.keras.applications.xception import preprocess_input
import numpy as np

image_model = tf.keras.applications.Xception(include_top=False, weights='imagenet')

def extract_xcepation_features(img_path):
    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(299, 299))  # Xception expects 299x299
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)   # [1, 299, 299, 3]
    x = preprocess_input(x)

    # Extract features
    features = image_model(x)   # shape: [1, 10, 10, 2048]
    features = tf.reshape(features, (1, 100, 2048))  # flatten spatial dims
    return features.numpy()


from ultralytics import YOLO
import logging
logging.getLogger('ultralytics').setLevel(logging.WARNING)
yolo_model = YOLO("yolo11n.pt")
input_dim = 2048

def extract_yolo_features(img):
    result = yolo_model(img)
    result = result[0]
    yolo_features = []
    for j, box in enumerate(result.boxes):
        xmin = result.boxes.xyxy[j][0].item()
        ymin = result.boxes.xyxy[j][1].item()
        xmax = result.boxes.xyxy[j][2].item()
        ymax = result.boxes.xyxy[j][3].item()
        confidence = result.boxes.conf[j].item()
        clss = result.boxes.cls[j].item()
        w = xmax - xmin
        h = ymax - ymin
        importance_factor = confidence * w * h
        # Append the YOLO features for this box
        yolo_features.append([xmin, ymin, w, h, confidence, clss, importance_factor])

    quickSort(yolo_features, 0, len(result.boxes)-1)
    yolo_features = np.array(yolo_features)
    yolo_features = np.array(yolo_features.flatten())
    yolo_features = np.pad(yolo_features, (0, input_dim - yolo_features.shape[0]), 'constant', constant_values=(0, 0)).astype(np.float32)
    return yolo_features


for i, (img_name) in enumerate(dataset):
    print(f"{i}/{dataset_len}")
    img = Image.open(img_name)

    # Extract yolo features
    yolo_features = extract_yolo_features(img)

    # Extract cnn features
    cnn_features = extract_xcepation_features(img_name)
    cnn_features = torch.from_numpy(cnn_features)
    cnn_features = cnn_features.squeeze(0) # [100, 2048]

    # Combine cnn and yolo features
    image_features = np.vstack((cnn_features.cpu().detach().numpy(), yolo_features)).astype(np.float32)
    np.save(img_name + "_" + features_type, image_features)