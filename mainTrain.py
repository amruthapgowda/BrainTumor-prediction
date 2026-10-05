import cv2
import os
import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical

image_directory = 'datasets/'

no_tumor_images = os.listdir(image_directory + 'no/')
yes_tumor_images = os.listdir(image_directory + 'yes/')

dataset = []
label = []

INPUT_SIZE = 64

# No Tumor Images
for image_name in no_tumor_images:

    if image_name.endswith('.jpg'):

        image = cv2.imread(image_directory + 'no/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))

        dataset.append(np.array(image))
        label.append(0)

# Tumor Images
for image_name in yes_tumor_images:

    if image_name.endswith('.jpg'):

        image = cv2.imread(image_directory + 'yes/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))

        dataset.append(np.array(image))
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)

print("No Tumor Images:", len(no_tumor_images))
print("Tumor Images:", len(yes_tumor_images))

# Split Data
x_train, x_test, y_train, y_test = train_test_split(
    dataset,
    label,
    test_size=0.2,
    random_state=0,

    shuffle=True
)

# Normalize
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One Hot Encoding
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# CNN Model
model = Sequential()

# Layer 1
model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 2
model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 3
model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Dense Layers
model.add(Flatten())

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(2))
model.add(Activation('softmax'))

# Compile
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train
history = model.fit(
    x_train,
    y_train,
    batch_size=16,
    epochs=50,
    validation_data=(x_test, y_test),
    
    shuffle=True,
    verbose=1
)

# Save Model
model.save('BrainTumor10EpochsCategorical.h5')

print("\nFinal Training Accuracy:",
      history.history['accuracy'][-1])

print("Final Validation Accuracy:",
      history.history['val_accuracy'][-1])