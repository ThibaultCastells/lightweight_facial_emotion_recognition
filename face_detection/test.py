#!/usr/bin/env python

"""
To run this script:
python test.py
"""

from PIL import Image
from src import detect_faces, show_bboxes
import os

root = "../data/test/"
input_dir = root+"img/"
output_dir = root+"img_with_bndbox/"

images = [f for f in os.listdir(input_dir) if f.lower().endswith('.jpg')]
for image_name in images:
    image = Image.open(input_dir+image_name)
    bd_box, ldmark = detect_faces(image)
    output_img = show_bboxes(image, bd_box, ldmark)
    output_img.save(output_dir+image_name)
    image.close()


# for image_name in images[0:20]:
#     image = Image.open(input_dir+image_name)
#     bouding_boxes, ldmark = detect_faces(image)
#     i = 0
#     for bndbox in bouding_boxes:
#         # remark: this method isn't ideal
#         # A better method would:
#         # - detect landmarks one the faces
#         # - use the landmarks to align the faces
#         face = image.crop(bndbox[0:4]).resize((100,100))
#         i+=1
#         face.save(output_dir+f"000{i}_"+image_name)
