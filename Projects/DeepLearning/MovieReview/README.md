# MovieReview

A sentiment analysis project comparing classical ML and deep learning approaches on movie reviews.

## Overview

This project demonstrates:
- Loading the IMDB movie review dataset
- Text decoding and exploration of sample reviews
- TF-IDF vectorization with n-grams
- Logistic Regression classification
- LSTM-based deep learning classification
- Evaluation with accuracy, ROC curve, and confusion matrices
- Predicting sentiment for custom review text

## Files

- `test.py` — main script that trains and evaluates Logistic Regression and LSTM models
- `class_distribution.png` — class balance visualization for the training set
- `roc_curve_lr.png` — ROC curve for the Logistic Regression model
- `lstm_curves.png` — training accuracy and loss curves for the LSTM
- `confusion_lstm.png` — confusion matrix for LSTM predictions

## Requirements

- Python 3.8+
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow

Install dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

## Run

```bash
python test.py
```

The script will train both models, print results, save plots, and demonstrate sentiment prediction on example text.

## Notes

- The dataset is loaded from `tensorflow.keras.datasets.imdb`.
- The LSTM model uses padded sequences of length 200 and an embedding layer.
- Logistic Regression is trained on TF-IDF features extracted from the review text.
