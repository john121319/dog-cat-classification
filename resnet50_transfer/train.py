import tensorflow as tf

from model import build_model


IMG_SIZE = (224, 224)


train_ds = tf.keras.utils.image_dataset_from_directory(
    "../data/train",
    image_size=IMG_SIZE,
    batch_size=32,
    color_mode="rgb",
    label_mode="binary"
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "../data/test",
    image_size=IMG_SIZE,
    batch_size=32,
    color_mode="rgb",
    label_mode="binary"
)


def preprocess(image, label):
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image, label


train_ds = train_ds.map(preprocess)
test_ds = test_ds.map(preprocess)

model = build_model()

model.summary()

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=5
)

model.save(
    "../results/resnet50_cats_dogs_model.keras"
)