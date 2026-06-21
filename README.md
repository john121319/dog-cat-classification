# Custom CNN vs. ResNet50 Transfer Learning for Cat/Dog Classification

I built this repository to explore and compare different approaches to binary image classification: training a custom Convolutional Neural Network from scratch, using a pre-trained ResNet50 model through transfer learning, improving it with data augmentation, and fine-tuning the model for better performance.

The goal of this project is to understand a complete deep learning computer vision workflow, including data cleaning, CNN model development, transfer learning, fine-tuning, model evaluation, and Grad-CAM explainability.

---

## 1. Project Directory Layout

I structured this codebase to keep data processing, baseline CNN training, transfer learning experiments, fine-tuning, evaluation scripts, and explainability outputs separated. This makes the project cleaner, easier to maintain, and more suitable for a GitHub portfolio.

```text
dog-cat-classification/
├── .gitignore
├── README.md
├── requirements.txt
│
├── evaluation/
│   ├── evaluate_models.py
│
├── data/
│   ├── train/
│   └── test/
│
├── preprocessing/
│   └── clean_dataset.py
│
├── cnn_baseline/
│   ├── model.py
│   ├── train.py
│   └── predict.py
│
├── resnet50_transfer/
│   ├── model.py
│   ├── train.py
│   └── predict.py
│
├── resnet50_transfer_with_augmentation/
│   ├── model.py
│   ├── train.py
│   ├── fine_tune.py
│   ├── predict.py
│   └── gradcam.py
│
└── results/
    ├── cnn_cats_dogs_model.keras
    ├── resnet50_cats_dogs_model.keras
    ├── resnet50_augmented_best_model.keras
    ├── resnet50_finetuned_model.keras
    ├── cnn_baseline_confusion_matrix.png
    ├── resnet50_transfer_confusion_matrix.png
    ├── gradcam_original.png
    ├── gradcam_heatmap.png
    └── gradcam_output.png
```

### Note

The raw dataset and trained `.keras` model files are stored locally and ignored using `.gitignore` to keep the public repository lightweight.

---

## 2. Dealing with Messy Data

Public image datasets often contain corrupted, unreadable, grayscale, or non-RGB images. These issues can cause TensorFlow training to crash during image loading or preprocessing.

### source of dataset
**[Download the Cats and Dogs Classification Dataset on Kaggle](https://kaggle.com)**

### Cleaning Script

I created:

```text
preprocessing/clean_dataset.py
```

### What the Cleaning Script Does

The script uses the `Pillow` library to:

1. Convert images into standard RGB format.
2. Remove corrupted or unreadable image files.
3. Reduce TensorFlow training errors caused by invalid JPEG files or unsupported image channels.

This preprocessing step helped make the dataset stable before training the CNN and ResNet50 models.

---

## 3. Model Configurations

### Approach A: Custom CNN Baseline

The first model is a custom CNN built from scratch. It uses three convolutional blocks followed by a dense classifier.

```text
Input (128x128x3)
        ↓
Conv2D(32) + MaxPooling
        ↓
Conv2D(64) + MaxPooling
        ↓
Conv2D(128) + MaxPooling
        ↓
Flatten
        ↓
Dense(128) + Dropout(0.5)
        ↓
Dense(1, Sigmoid)
```

This model learns visual features directly from the cat and dog dataset.

---

### Approach B: ResNet50 Transfer Learning

The second model uses ResNet50, a deep convolutional neural network pre-trained on ImageNet.

Instead of training all convolutional layers from scratch, I used ResNet50 as a frozen feature extractor and added a custom classification head.

```text
Input (224x224x3)
        ↓
ResNet50 Pretrained Backbone (Frozen)
        ↓
GlobalAveragePooling2D
        ↓
Dense(128) + Dropout(0.5)
        ↓
Dense(1, Sigmoid)
```

This approach allows the model to reuse powerful visual features learned from a large image dataset.

---

### Approach C: ResNet50 with Data Augmentation and Fine-Tuning

The final model improves the transfer learning approach by adding data augmentation and fine-tuning selected ResNet50 layers.

```text
Input (224x224x3)
        ↓
Data Augmentation
        ↓
ResNet50 Preprocessing
        ↓
ResNet50 Backbone
        ↓
GlobalAveragePooling2D
        ↓
Dense(128) + Dropout(0.5)
        ↓
Dense(1, Sigmoid)
```

After training the classification head, I fine-tuned the final ResNet50 layers using a very small learning rate. This helped the model adapt pretrained ImageNet features to the cat/dog classification task.

---

## 4. Model Evaluation and Performance

I evaluated the models using an independent test dataset of 2,495 images, evenly distributed between cats and dogs.

### Performance Summary

| Model Architecture         | Test Accuracy | Test Loss |
| :------------------------- | :-----------: | :-------: |
| CNN Baseline               |     83.41%    |   0.5558  |
| ResNet50 Transfer Learning |     99.12%    |   0.0229  |
| ResNet50 Fine-Tuned        |     99.36%    |   0.0155  |

### Key Findings

* The custom CNN baseline achieved good beginner-level performance, but it struggled compared with the pretrained model.
* ResNet50 transfer learning produced a major improvement, increasing accuracy from 83.41% to 99.12%.
* Fine-tuning improved the ResNet50 model further, increasing accuracy from 99.12% to 99.36%.
* The loss decreased from 0.0229 to 0.0155 after fine-tuning, showing that the fine-tuned model made more confident and better-calibrated predictions.
* The best-performing model was the fine-tuned ResNet50 model.

---

## 5. Detailed Evaluation Results

I used `evaluate_models.py` to calculate precision, recall, F1-score, and confusion matrices.

### CNN Baseline Classification Report

| Class | Precision | Recall | F1-score | Support |
| ----- | --------- | ------ | -------- | ------- |
| Cats  | 0.81      | 0.87   | 0.84     | 1248    |
| Dogs  | 0.86      | 0.80   | 0.83     | 1247    |

### CNN Baseline Accuracy

```text
83.41%
```

The CNN baseline performed reasonably well, but it made more mistakes compared with the ResNet50 models.

---

### ResNet50 Transfer Learning Classification Report

| Class | Precision | Recall | F1-score | Support |
| ----- | --------- | ------ | -------- | ------- |
| Cats  | 0.99      | 0.99   | 0.99     | 1248    |
| Dogs  | 0.99      | 0.99   | 0.99     | 1247    |

### ResNet50 Transfer Learning Accuracy

```text
99.12%
```

The ResNet50 transfer learning model achieved very strong performance with high precision, recall, and F1-score for both classes.

---

## 6. Grad-CAM Explainability

I used Grad-CAM to visualize which image regions influenced the ResNet50 model's predictions. This helps interpret whether the model is focusing on meaningful animal features such as the face, ears, mouth, and body instead of irrelevant background patterns.

### Grad-CAM Output Files

The Grad-CAM outputs are saved in:

```text
results/gradcam_original.png
results/gradcam_heatmap.png
results/gradcam_output.png
```

### Why Grad-CAM Matters

Grad-CAM is especially important for future medical AI work because it can help verify whether a model is focusing on medically meaningful regions instead of artifacts, labels, or image borders.

---

## 7. Setting Up and Running the Code

This project was built and tested on macOS Apple Silicon using a Conda Python environment.

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 8. Order of Execution

### Step 1: Clean the Dataset

Run this command from the main project folder:

```bash
python preprocessing/clean_dataset.py
```

### Step 2: Train the Custom CNN Baseline

```bash
cd cnn_baseline
python train.py
cd ..
```

### Step 3: Train the ResNet50 Transfer Learning Model

```bash
cd resnet50_transfer
python train.py
cd ..
```

### Step 4: Train ResNet50 with Augmentation

```bash
cd resnet50_transfer_with_augmentation
python train.py
cd ..
```

### Step 5: Fine-Tune ResNet50

```bash
cd resnet50_transfer_with_augmentation
python fine_tune.py
cd ..
```

### Step 6: Compare Model Accuracy and Loss

```bash
cd evaluation
python evaluate_models.py
```

### Step 7: Generate Detailed Evaluation Reports

```bash
python evaluate_models.py
```

This script prints:

```text
Accuracy
Loss
Precision
Recall
F1-score
Confusion Matrix
```

It also saves confusion matrix images inside the `results/` folder.

### Step 8: Generate Grad-CAM Visualization

```bash
cd resnet50_transfer_with_augmentation
python gradcam.py
cd ..
```

This saves:

```text
results/gradcam_original.png
results/gradcam_heatmap.png
results/gradcam_output.png
```

---

## 9. Key Learning Outcomes

Through this project, I learned how to:

* Build a CNN model from scratch.
* Load image datasets using TensorFlow.
* Clean corrupted image files before training.
* Train and evaluate a binary image classification model.
* Apply ResNet50 transfer learning.
* Use data augmentation to improve generalization.
* Fine-tune pretrained ResNet50 layers.
* Compare CNN, transfer learning, and fine-tuned models.
* Use precision, recall, F1-score, and confusion matrices for deeper evaluation.
* Apply Grad-CAM to visualize model attention.

---

## 10. Conclusion

The custom CNN baseline achieved good performance, but the ResNet50 transfer learning model performed much better. Fine-tuning further improved performance and reduced the model loss.

This experiment shows that transfer learning and fine-tuning are powerful techniques for image classification tasks, especially when using pretrained models that have already learned general visual features from large datasets.

This project also provides a strong foundation for future work in medical image classification, explainable AI, and federated learning.

---

## Author

Yohannes A
