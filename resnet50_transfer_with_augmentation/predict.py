import tensorflow as tf
import numpy as np


MODEL_PATH = "../results/resnet50_augmented_best_model.keras"
IMAGE_PATH = "../test_images/dog1.jpg"


model = tf.keras.models.load_model(MODEL_PATH)


img = tf.keras.utils.load_img(
    IMAGE_PATH,
    target_size=(224, 224)
)


img = tf.keras.utils.img_to_array(img)


img = np.expand_dims(img, axis=0)


prediction = model.predict(img)

probability = prediction[0][0]

print("Prediction probability:", probability)


if probability > 0.5:
    print("Dog")
else:
    print("Cat")