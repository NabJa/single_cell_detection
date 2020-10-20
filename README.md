# Single cell detection

This repository contains all tools to train and test the object detection algorithms Faster RCNN and SSD.
Implementation of the networks can be found in the tensorflow object detection API.
Evaluation is done on both, brightfield and lensfree microscopy images.


## Training
Tensorflow version: < 2.0  
To be able to train your own network the tensorflow object detection API must be installed. Make sure that the API is apped to your path variable. 
 

Training scripts:  
- [Train a single model](training/train.py) (<- Change path vairables to our own path!)
- [Train multiple models](training/train_all_models.py)

## Prediction
Tensorflow version: >= 2.0  
Instruction: [wiki](https://gitlab.lrz.de/single_cell_heterogeneity/single_cell_detection/-/wikis/Prediction)

## Evaluation
- Model evaluation: [LF_vs_BF_evaluation.ipynb](LF_vs_BF_evaluation.ipynb)
- Generalizability on unseen cell types: [cell_type_generalization.ipynb](cell_type_generalization.ipynb)

For command line evaluation use [evaluate.py](evaluate.py).
