# MINST Data

A convolutional neural network project for handwritten digit classification using the MNIST dataset.

## Overview

This project demonstrates:
- Loading and visualizing MNIST image data
- Preprocessing images for CNN input
- Building a Keras CNN model for digit classification
- Training with validation split
- Evaluating model accuracy and loss
- Visualizing training curves and confusion matrix
- Saving the trained model to `mnist_cnn_model.h5`

## Files

- `test.py` — main script that builds, trains, evaluates, and saves the CNN model
- `mnist_samples.png` — sample handwritten digit images from the training set
- `training_curves.png` — model accuracy and loss curves over training epochs
- `confusion_matrix.png` — confusion matrix for test set predictions
- `predictions.png` — sample predictions with correct and incorrect classifications
- `mnist_cnn_model.h5` — saved Keras model file

## Requirements

- Python 3.8+
- numpy
- matplotlib
- seaborn
- tensorflow

Install dependencies with:

```bash
pip install numpy matplotlib seaborn tensorflow
```

## Run

```bash
python test.py
```

The script will train a CNN on MNIST, display evaluation results, create plots, and save the trained model.

## Notes

- The model uses three convolutional layers followed by a dense classifier.
- Training is performed for 10 epochs with a 10% validation split.
- The dataset is loaded from `tensorflow.keras.datasets.mnist`.
