# Light-weight facial emotion recognition

A light network to detect faces and classify emotions out of 5 categories (neutral, anger, surprise, smile and sad).

The constraint in this project was to obtain a model of size <100MB (the lighter the better), while maintaining a good accuracy.

## How to use it

To train a new model, choose you parameters in *train.py* and run the following command:
```bash
python train.py --data_path "path/to/your/data/"
```
The models are saved in *emotion_recognition/models*.

To test an existing model, run the following command:
```bash
python main.py --data_path "path/to/your/data/" --model_name "model_name"
```
The results are saved in xml files, at *data_path/test/annotations*.

## Requirements
* pytorch 1.5
* Pillow, numpy

## Model size

A model generated with this method weight around 241Ko.
If we add to this the weight of the model for face detection  (~2.9Mo), we get a total sligtly higher than 3Mo.

## Futur improvements

- To improve the accuracy of the emotion recognition, it could be interesting to use a face alignment method during the pre-processing. However, this would necessite to add a network, which would increase the global models size.
- The dataset used to train the models is small, it contains around 2000 faces. Data transformation partially solves this problem, but a bigger dataset would probably have a strong positive impact on the performance of the model.
- To reduce the size even more, it is possible to use model compression methods (pruning, quantization, ...).


