import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers
import tensorflow as tf

# Paramètres
batch_size = 16
img_size = (64, 64)
CHANNELS = 3
EPOCHS = 20

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Chemins (à adapter selon ton PC)
train_ds = datagen.flow_from_directory(
    directory='/path/to/your/dataset/train',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Modèle CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(16, (3,3), activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(8, (3,3), activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=['Recall']
)

# Entraînement
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    verbose=1
)

# Sauvegarde du modèle
model.save('model/cnn_model.h5')
print("Modèle entraîné et sauvegardé avec succès !")