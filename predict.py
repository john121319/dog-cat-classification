import tensorflow as tf

import numpy as np



model = tf.keras.models.load_model(

    "cats_dogs_model.keras"

)



img = tf.keras.utils.load_img(

    "test_images/dog1.jpg",

    target_size=(128,128)

)



img = tf.keras.utils.img_to_array(

    img

)



img = img/255.0



img = np.expand_dims(

    img,

    axis=0

)



prediction = model.predict(

    img

)



if prediction[0][0] > 0.5:

    print("Dog")

else:

    print("Cat")