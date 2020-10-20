# Single cell detection

This repository contains all tools to train and test the object detection algorithms Faster RCNN and SSD.
Implementation of the networks can be found in the tensorflow object detection API.
Evaluation is done on both, brightfield and lensfree microscopy images.


## Prediction
Tensorflow version: >= 2.0  
Instruction: [wiki](https://gitlab.lrz.de/single_cell_heterogeneity/single_cell_detection/-/wikis/Prediction)

## Training
Tensorflow version: < 2.0  
To be able to train your own network the tensorflow object detection API must be installed. Make sure that the API is apped to your path variable. 
 

Training scripts:  
- [Train a single model](training/train.py) (<- Change path vairables to our own path!)
- [Train multiple models](training/train_all_models.py)


## Evaluation
Evaluation and exploration of the trained models can be done in the following notebooks:
- Evaluation of a single model: [detector_evaluation.ipynb](detector_evaluation.ipynb)
- Comparison of multiple models: [Evaluation_of_multiple_models.ipynb](Evaluation_of_multiple_models.ipynb)
- Generalizability on unseen cell types: [cell_type_generalization.ipynb](cell_type_generalization.ipynb)

For command line evaluation use [evaluate.py](evaluate.py).
