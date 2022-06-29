# anatomical-structures-classifier

## A deep learning project
- Convolutional neural network made with [PyTorch](https://pytorch.org) for multi-class classification of medical data.
- The dataset is composed of 20 CT scans images from 20 patients. There are 23 classes (anatomicals stuctures sush as muscles, bones and organs). Dataset is not shared.
- A cross-validation is used with 4 folds (15 training patients, 5 testing patients).
- A confusion matrix is computed and saved and accuracies / losses of each epoch (training and testing) are saved.
- Trained models are saved for [anatomical structures detection](https://github.com/NicolasLvrd/anatomical-structures-detector).

## Context
- This implementation is part of a research initiation project at [INSA Lyon](https://www.insa-lyon.fr) (1st year of telecom engineering) under supervision of researcher from [CREATIS](https://www.creatis.insa-lyon.fr/site7/fr) laboratory.
- This CNN model is only usable with a specific dataset. Therefore it can't be used as such.
