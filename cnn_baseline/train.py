import tensorflow as tf

from model import build_model


train_ds = tf.keras.utils.image_dataset_from_directory(
    "../data/train",
    image_size=(128, 128),
    batch_size=32,
    color_mode="rgb",
    label_mode="binary"
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "../data/test",
    image_size=(128, 128),
    batch_size=32,
    color_mode="rgb",
    label_mode="binary"
)


normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(
    lambda x, y: (normalization_layer(x), y)
)

test_ds = test_ds.map(
    lambda x, y: (normalization_layer(x), y)
)

model = build_model()

model.summary()

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=10
)

model.save(
    "../results/cnn_cats_dogs_model.keras"
)