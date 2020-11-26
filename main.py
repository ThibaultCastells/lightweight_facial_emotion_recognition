#!/usr/bin/env python

"""
To run this script:
python main.py --data_path "path/to/your/data/" --model_name "model_name"
Example: python main.py --data_path "data/" --model_name "model_48_76.model"

File to run a test.
Pipeline:
1) detect faces in the test set and save the bouding boxes in xml files
2) create a dataset of faces from these bounding boxes
3) for each face, recognise the emotion and update the associated xml file
"""

import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import argparse
import os

import torch
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from face_detection.detect_faces import save_faces
from emotion_recognition.datasets.faces_dataset import FacesDataset
from emotion_recognition.archs.model import Model

# args
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--model_name", type=str, required=True)
args = parser.parse_args()

# method to extract faces from the images in the given dataset (generate xml files):
save_faces(os.path.join(args.data_path, "test"))

# load network
net = Model(num_classes=5)
checkpoint = torch.load(os.path.join('emotion_recognition/models', args.model_name), map_location=torch.device('cpu'))
net.load_state_dict(checkpoint['net'])

# transforms
transform_test = transforms.Compose([
    transforms.ToTensor()
])

# intialisation dataset
shape = (50,50)
data = FacesDataset(False, args.data_path, shape=shape, transform=transform_test)
data_loader = DataLoader(data, shuffle=False)

for x_test, xml_file, index in data_loader:
    y_test = np.argmax(net(x_test).detach().numpy())

    # update of the xml file associated to this face
    xml_path = os.path.join(args.data_path, f"test/annotations/{xml_file[0]}")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    idx = -1
    for i in range(len(root)):
        if root[i].tag=="object":
            idx+=1
            if idx == index:
                root[i][0].text = data.classes[y_test]
                break
    tree.write(xml_path)
