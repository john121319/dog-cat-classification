import tensorflow as tf

from model import build_model



train_ds = tf.keras.utils.image_dataset_from_directory(

    "dogs-vs-cats-classification/train",

    image_size=(128,128),

    batch_size=32

)



test_ds = tf.keras.utils.image_dataset_from_directory(

    "dogs-vs-cats-classification/test",

    image_size=(128,128),

    batch_size=32

)



normalization_layer = tf.keras.layers.Rescaling(

    1./255

)



train_ds = train_ds.map(

    lambda x,y:

    (normalization_layer(x),y)

)



test_ds = test_ds.map(

    lambda x,y:

    (normalization_layer(x),y)

)




model = build_model()



model.summary()



history = model.fit(

    train_ds,

    validation_data=test_ds,

    epochs=10

)



model.save(

    "cats_dogs_model.keras"

)