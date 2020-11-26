#!/usr/bin/env python

from PIL import Image
# from src.visualization_utils import show_bboxes
# from src.detector import detect_faces
import xml.etree.ElementTree as ET
from xml.etree import ElementTree
from xml.dom import minidom
import os

def save_faces(data_directory):
    """ Create a xml file for each image, in which the bouding boxes of the faces
        in the image are saved.
    Args:
        - data_directory: path to the data directory
    """

    input_dir = os.path.join(data_directory, "img")
    output_dir = os.path.join(data_directory, "annotations")
    if not os.path.isdir(input_dir): # make sure that the img file exists
        raise NotADirectoryError(f"{input_dir} doesn't exist")
    if not os.path.isdir(output_dir): # make the annotation file
        os.mkdir(output_dir)


    images = [f for f in os.listdir(input_dir) if f.lower().endswith('.jpg')]
    for image_name in images:
        xml_name = os.path.splitext(image_name)[0]+".xml"
        xml_path = os.path.join(output_dir, xml_name)

        # if the image as already been processed, it is ignored
        if os.path.exists(xml_path):
            continue
        # else, the faces are extracted
        image = Image.open(input_dir+image_name)
        w, h = image.size
        bnd_boxes = detect_faces(image)[0]
        image.close()

        # create the file structure
        annotation = ET.Element('annotation')
        filename = ET.SubElement(annotation, 'filename')
        filename.text = image_name
        size = ET.SubElement(annotation, 'size')
        width = ET.SubElement(size, 'width')
        height = ET.SubElement(size, 'height')
        width.text = str(w)
        height.text = str(h)
        for face_bnd_box in bnd_boxes:
            object = ET.SubElement(annotation, 'object')
            name = ET.SubElement(object, 'name')
            name.text = "Unknown"
            bndbox = ET.SubElement(object, 'bndbox')
            xmin = ET.SubElement(bndbox, 'xmin')
            xmin.text = str(int(face_bnd_box[0]))
            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = str(int(face_bnd_box[1]))
            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = str(int(face_bnd_box[2]))
            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = str(int(face_bnd_box[3]))

        # create a new XML file with the results
        image_annotations = ET.tostring(annotation, 'utf-8')
        myfile = open(xml_path, "w")

        image_annotations = minidom.parseString(image_annotations)
        image_annotations = image_annotations.toprettyxml(indent="  ")
        myfile.write(image_annotations)
        myfile.close()

if __name__ == '__main__':
    path = "../data/test/"
    save_faces(path)
