import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay


TEST_PATH = "data/test"

CNN_MODEL_PATH = "results/cnn_cats_dogs_model.keras"
RESNET_MODEL_PATH = "results/resnet50_cats_dogs_model.keras"

BATCH_SIZE = 32

CNN_IMAGE_SIZE = (128, 128)
RESNET_IMAGE_SIZE = (224, 224)


os.makedirs("results", exist_ok=True)


def load_cnn_test_data():

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_PATH,
        image_size=CNN_IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        label_mode="binary",
        shuffle=False
    )

    class_names = test_ds.class_names

    test_ds = test_ds.map(
        lambda x, y: (x / 255.0, y)
    )

    return test_ds, class_names


def load_resnet_test_data():

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_PATH,
        image_size=RESNET_IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        label_mode="binary",
        shuffle=False
    )

    class_names = test_ds.class_names

    def preprocess(image, label):
        image = tf.keras.applications.resnet50.preprocess_input(image)
        return image, label

    test_ds = test_ds.map(preprocess)

    return test_ds, class_names


def get_true_labels(dataset):

    labels = []

    for images, batch_labels in dataset:
        labels.extend(batch_labels.numpy())

    labels = np.array(labels).reshape(-1)

    return labels.astype(int)


def get_predictions(model, dataset):

    probabilities = model.predict(dataset)

    probabilities = probabilities.reshape(-1)

    predictions = (probabilities > 0.5).astype(int)

    return predictions


def evaluate_model(model_path, dataset, class_names, model_name):

    print("\n=================================")
    print(model_name)
    print("=================================")

    model = tf.keras.models.load_model(model_path)

    loss, accuracy = model.evaluate(dataset)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Loss: {loss:.4f}")

    y_true = get_true_labels(dataset)

    y_pred = get_predictions(model, dataset)

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names
        )
    )

    cm = confusion_matrix(y_true, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(values_format="d")

    plt.title(f"{model_name} Confusion Matrix")

    save_path = f"results/{model_name.lower().replace(' ', '_')}_confusion_matrix.png"

    plt.savefig(save_path)

    plt.close()

    print(f"Confusion matrix saved to: {save_path}")


cnn_test_ds, class_names = load_cnn_test_data()

resnet_test_ds, _ = load_resnet_test_data()


evaluate_model(
    CNN_MODEL_PATH,
    cnn_test_ds,
    class_names,
    "CNN Baseline"
)


evaluate_model(
    RESNET_MODEL_PATH,
    resnet_test_ds,
    class_names,
    "ResNet50 Transfer"
)