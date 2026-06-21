import os
import tensorflow as tf
from model import build_model


TRAIN_PATH = "../data/train"
TEST_PATH = "../data/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

os.makedirs("../results", exist_ok=True)


train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    label_mode="binary",
    shuffle=True,
    seed=123
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    label_mode="binary",
    shuffle=False
)

print("Class names:", train_ds.class_names)


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


model = build_model()

model.summary()


early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "../results/resnet50_augmented_best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)


history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)


model.save("../results/resnet50_augmented_final_model.keras")