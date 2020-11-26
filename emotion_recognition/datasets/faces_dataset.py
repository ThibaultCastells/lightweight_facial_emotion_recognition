#!/usr/bin/env python
import torchvision
from PIL import Image
import numpy as np
from xml.dom import minidom
import os

class FacesDataset:
    """ FacesDataset data set
    Args:
        train: True if training, else False
        root_data: path to the data directory
    """
    def __init__(self, train, root_data, transform=None, shape=(100,100), **kw):
        self.train = train
        self.classes = ['neutral', 'smile', 'sad', 'anger', 'surprise']
        self.n_classes = len(self.classes)
        self.transform = transform
        self.shape=shape

        path = os.path.join(root_data, 'train') if train else os.path.join(root_data, 'test')
        path_annot = os.path.join(path, 'annotations') # annotations directory
        path_image = os.path.join(path, 'img') # images directory

        self.db = [] # dataset is [(face, label)] (train) or [(face, xml_filename, index_in_file)] (test)
        files = [f for f in os.listdir(path_image) if f.lower().endswith('.jpg')]
        for image_name in files:
            # - collect the associated metadata (xml file)
            xml_name = os.path.splitext(image_name)[0]+".xml"
            bouding_boxes, labels = self.get_annotations(os.path.join(path_annot, xml_name))

            # - extract the faces and resize them
            image = Image.open(os.path.join(path_image, image_name))
            image.load()
            faces = self.extract_and_resize(image, bouding_boxes)
            # - for each resized face, add it in self.db (with its label if train=True)
            for idx in range(len(faces)):
                if train:
                    self.db.append((faces[idx], labels[idx]))
                else:
                    self.db.append((faces[idx], xml_name, idx))

    def get_annotations(self, path):
        """ function to get the bouding boxes (and labels if self.train=True)
        Args:
            path: path to the xml file
        Return:
            - a list of bounding boxes, one for each face referenced in the xml file
            - the label associated to these faces if self.train=True, else an empty array
        """
        bouding_boxes = []
        labels = []

        xml_file = minidom.parse(path)
        faces = xml_file.getElementsByTagName('object')
        for face in faces:
            xmin = int(face.getElementsByTagName("xmin")[0].firstChild.data)
            ymin = int(face.getElementsByTagName("ymin")[0].firstChild.data)
            xmax = int(face.getElementsByTagName("xmax")[0].firstChild.data)
            ymax = int(face.getElementsByTagName("ymax")[0].firstChild.data)
            bouding_boxes.append([xmin,ymin,xmax,ymax])
            if self.train:
                label = face.getElementsByTagName('name')[0].firstChild.data
                if label not in self.classes:
                    raise ValueError(f"{label} isn't a valid class")
                labels.append(self.classes.index(label))
        return bouding_boxes, labels

    def extract_and_resize(self, image, bouding_boxes):
        """ function to extract faces from a given image, and resize them
        Args:
            image: an Image object
            bouding_boxes: a list of bouding boxes [xmin,ymin,xmax,ymax], one
                for each face in the image.
        Return: a list of Images, corresponding to the faces in the input image (resized)
        """
        faces = []
        for bndbox in bouding_boxes:
            # remark: this method isn't ideal
            # A better method would:
            # - detect landmarks one the faces
            # - use the landmarks to align the faces
            face = image.crop(bndbox).resize(self.shape).convert('L')
            faces.append(face)

        return faces


    def __len__(self):
        return len(self.db)

    def __repr__(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

    def __getitem__(self, idx):
        if self.train:
            image, emotion = self.db[idx]
            if self.transform is not None:
                image = self.transform(image)
            return image, emotion
        else:
            image, xml_file, index = self.db[idx]
            if self.transform is not None:
                image = self.transform(image)
            return image, xml_file, index



if __name__ == '__main__':
    path = "../../data/"
    data = FacesDataset(True, path)
    print(data)
    for i in range(3):
        print(f"data[{i}]: {data[i]}\n--")
