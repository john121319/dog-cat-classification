import tensorflow as tf


IMG_SIZE_CNN = (128, 128)
IMG_SIZE_RESNET = (224, 224)
BATCH_SIZE = 32
TEST_PATH = "data/test"


# =========================
# Load test data for CNN
# =========================
cnn_test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=IMG_SIZE_CNN,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    label_mode="binary"
)

cnn_test_ds = cnn_test_ds.map(
    lambda x, y: (x / 255.0, y)
)


# =========================
# Load test data for ResNet50
# =========================
resnet_test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=IMG_SIZE_RESNET,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    label_mode="binary"
)


def resnet_preprocess(image, label):
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image, label


resnet_test_ds = resnet_test_ds.map(resnet_preprocess)


# =========================
# Load models
# =========================
cnn_model = tf.keras.models.load_model(
    "results/cnn_cats_dogs_model.keras"
)

resnet_model = tf.keras.models.load_model(
    "results/resnet50_cats_dogs_model.keras"
)


# =========================
# Evaluate models
# =========================
cnn_loss, cnn_acc = cnn_model.evaluate(cnn_test_ds)
resnet_loss, resnet_acc = resnet_model.evaluate(resnet_test_ds)


# =========================
# Print comparison
# =========================
print("\nModel Comparison")
print("========================")
print(f"CNN Baseline Accuracy: {cnn_acc:.4f}")
print(f"CNN Baseline Loss:     {cnn_loss:.4f}")

print("------------------------")

print(f"ResNet50 Accuracy:     {resnet_acc:.4f}")
print(f"ResNet50 Loss:         {resnet_loss:.4f}")