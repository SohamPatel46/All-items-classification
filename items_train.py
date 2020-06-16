import os
from tensorflow import keras
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing.image import ImageDataGenerator

f=__file__
f=f.replace('items_train.py', '')
model_save=os.path.join(f)
root=f+'ROOM ITEMS'
root_dir = os.path.join(root)

TRAINING_DIR = root_dir
training_datagen = ImageDataGenerator(rescale = 1./255)
train_generator = training_datagen.flow_from_directory(TRAINING_DIR,	target_size=(150,150),	
                                                       class_mode='categorical',batch_size=2)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(loss = 'categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
history = model.fit(train_generator, epochs=10,  verbose = 1)
model.save(model_save+"model.h5")
