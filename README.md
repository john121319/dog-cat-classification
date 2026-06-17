# Cats vs Dogs Image Classification using CNN

A Convolutional Neural Network (CNN) project built with TensorFlow and Keras to classify images of cats and dogs.

---

## Project Overview

This project implements a CNN model from scratch for binary image classification.

The model is trained on the Kaggle Dogs vs Cats dataset and can classify an image as:

- Cat
- Dog

The project also includes a dataset cleaning script to remove corrupted or invalid images before training.

---

## Features

- CNN implemented from scratch
- TensorFlow / Keras
- Image normalization
- Dataset cleaning utility
- Model saving and loading
- Single image prediction
- Kaggle dataset support

---

## Project Structure

```text
cats-vs-dogs-cnn/

│

├── model.py

├── train.py

├── predict.py

├── clean_dataset.py

├── cats_dogs_model.keras

├── README.md

├── .gitignore

│

└── dogs-vs-cats-classification/

    ├── train/

    │   ├── cats/

    │   └── dogs/

    │

    └── test/

        ├── cats/

        └── dogs/
```

---

## Dataset

Dataset used:

**Dogs vs Cats Dataset**

Download from Kaggle:

https://www.kaggle.com/datasets/salader/dogs-vs-cats

---

## Dataset Cleaning

Some images in the dataset may be corrupted or have invalid channels.

The project includes:

```python
clean_dataset.py
```

which:

- checks all images
- removes corrupted images
- fixes dataset issues before training

---

## CNN Architecture

```text
Input (128x128x3)

↓

Conv2D(32, 3x3)

↓

MaxPooling2D

↓

Conv2D(64, 3x3)

↓

MaxPooling2D

↓

Conv2D(128, 3x3)

↓

MaxPooling2D

↓

Flatten

↓

Dense(128)

↓

Dropout(0.5)

↓

Dense(1, sigmoid)

↓

Cat / Dog
```

---

## Training

Run:

```bash
python train.py
```

Example result:

```text
Epoch 10/10

Train Accuracy:

92.5%

Validation Accuracy:

84.8%
```

---

## Prediction

Predict a single image:

```bash
python predict.py
```

Output:

```text
Prediction:

Dog

Confidence:

98%
```

---

## Technologies Used

- Python

- TensorFlow

- Keras

- NumPy

- Pillow

- Matplotlib

---

## Future Improvements

- Transfer Learning using ResNet50

- Data Augmentation

- Medical Image Classification

- Federated Learning

---

## Author

Yohannes A.

Machine Learning Enthusiast

Interested in Medical AI and Federated Learning.