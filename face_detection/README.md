# MTCNN

A modified version of  [the pytorch implementation](https://github.com/TropComplique/mtcnn-pytorch) of **inference stage** of face detection algorithm described in  
[Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/abs/1604.02878) 



## How to use it

Basic commands:
```python
from src import detect_faces
from PIL import Image

image = Image.open('path/to/image.jpg')
bounding_boxes, landmarks = detect_faces(image)
```

To run the script detect_faces.py:
```bash
python detect_faces.py
```

## Requirements
* pytorch 4.2
* Pillow, numpy
