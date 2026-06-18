# Custom CNN vs. ResNet50 Transfer Learning for Cat/Dog Classification

I built this repository to explore and compare two different approaches to binary image classification: training a custom Convolutional Neural Network from scratch versus leveraging a pre-trained ResNet50 model via transfer learning. 


---

## 1. Project Directory Layout

I structured this codebase to keep data processing, baseline testing, and transfer learning experiments strictly separated. This makes it easier to track training logs and evaluate scripts independently.

```text
dog-cat-classification/
├── .gitignore                   # Keeps heavy model weights (*.keras) and raw data out of Git
├── README.md                    # Project documentation
├── requirements.txt             # Exact package dependencies for environment replication
│
├── data/                        # Local folder for image datasets (ignored by Git)
│   ├── train/                   # Subfolders for cats/dogs training images
│   └── test/                    # Subfolders for cats/dogs validation images
│
├── preprocessing/               
│   └── clean_dataset.py         # Script to scrub corrupted or invalid images before training
│
├── cnn_baseline/                
│   ├── model.py                 # Custom layers implementation
│   ├── train.py                 # Core training loop and plotting logic
│   └── predict.py               # Quick single-image inference helper
│
├── resnet50_transfer/           
│   ├── model.py                 # ResNet50 feature extractor and dense head layout
│   ├── train.py                 # Freezes backbone and trains classification layer
│   └── predict.py               # Transfer learning model inference script
│
└── results/                     # Saved training curves and visualization dashboards
    ├── cnn_accuracy.png
    ├── resnet50_accuracy.png
    └── sample_predictions/
```

*Note: The raw data folders and compiled `*.keras` model weights are kept on my local iMac and masked via `.gitignore` to keep the public repository lightweight and clean.*

---

## 2. Dealing with Messy Data (Pre-flight Data Cleaning)

Public vision datasets frequently contain broken files that cause training loops to crash mid-epoch. Before feeding any images into either network, I wrote `preprocessing/clean_dataset.py` to scrub the source data.

The script runs a quick sweep through the directories using the `Pillow` library to:
1. **Enforce RGB Alignment:** Detect and convert any grayscale or 4-channel RGBA images into standard 3-channel RGB format.
2. **Filter Out Broken Binaries:** Scan and delete images with truncated file streams, corrupt headers, or incomplete JPEG sequences that throw errors during tensor loading.

---

## 3. Model Configurations

### Approach A: Custom CNN Baseline
A standard, lightweight feature extraction network. It uses three alternating blocks of 3x3 convolutions and 2x2 max-pooling to shrink spatial dimensions before feeding into a dense 128-neuron hidden layer with 50% Dropout to control overfitting.
```text
Input (128x128x3) ──► Conv2D(32) ──► MaxPool ──► Conv2D(64) ──► MaxPool ──► Conv2D(128) ──► MaxPool ──► Flatten ──► Dense(128) ──► Dropout(0.5) ──► Dense(1, Sigmoid)
```

### Approach B: ResNet50 Transfer Learning
To evaluate how well a model trained on millions of natural images transfers to this specific task, I used a pre-trained ResNet50 backbone (initialized with ImageNet weights) as a frozen feature extractor. I stacked a global average pooling layer and a custom dense classification head on top.
```text
Input (224x224x3) ──► ResNet50 (Frozen) ──► GlobalAvgPooling ──► Dense(128) ──► Dropout(0.5) ──► Dense(1, Sigmoid)
```

---

## 4. Model Comparison

| Model | Accuracy | Loss |
|------|----------|------|
| CNN Baseline | 83.41% | 0.5558 |
| ResNet50 Transfer Learning | 99.12% | 0.0229 |

The ResNet50 transfer learning model significantly outperformed the custom CNN baseline. This shows that pretrained deep learning models can extract more powerful image features and achieve higher accuracy with less training time.
---

## 5. Setting Up and Running the Code

### Environment Replication
This project was built and tested on an M1 Mac using a clean Python 3.10 environment to prevent library dependency conflicts and ensure native Apple Silicon GPU acceleration support.

```bash
# Set up a clean virtual workspace using Conda
conda create -n env_cats_dogs python=3.10 -y
conda activate env_cats_dogs

# Install required framework versions
pip install -r requirements.txt
```

### Order of Execution
1. Run the data sanitation script to purify your local image dataset:
   ```bash
   python preprocessing/clean_dataset.py
   ```
2. Train the baseline custom CNN model:
   ```bash
   cd cnn_baseline && python train.py
   ```
3. Train the transfer learning model:
   ```bash
   cd ../resnet50_transfer && python train.py
   ```
4. evaluate models:
   ```bash
   cd ../ python evaluate_models.py
   ```
---

**Author**  
Yohannes A 
