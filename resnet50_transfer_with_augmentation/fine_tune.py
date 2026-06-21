import os
import tensorflow as tf


TRAIN_PATH = "../data/train"
TEST_PATH = "../data/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
FINE_TUNE_EPOCHS = 5

BASE_MODEL_PATH = "../results/resnet50_augmented_best_model.keras"
FINE_TUNED_MODEL_PATH = "../results/resnet50_finetuned_model.keras"


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


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


model = tf.keras.models.load_model(BASE_MODEL_PATH)


base_model = None

for layer in model.layers:
    if isinstance(layer, tf.keras.Model) and "resnet50" in layer.name.lower():
        base_model = layer
        break


if base_model is None:
    raise ValueError("ResNet50 base model was not found inside the loaded model.")


base_model.trainable = True


for layer in base_model.layers[:-30]:
    layer.trainable = False


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


model.summary()


early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


checkpoint = tf.keras.callbacks.ModelCheckpoint(
    FINE_TUNED_MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)


history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=[early_stop, checkpoint]
)


loss, accuracy = model.evaluate(test_ds)

print("Fine-tuned model accuracy:", accuracy)
print("Fine-tuned model loss:", loss)


model.save(FINE_TUNED_MODEL_PATH)