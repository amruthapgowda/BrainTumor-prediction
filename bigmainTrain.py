import cv2
import os
import numpy as np
import tensorflow as tf
tf.keras.utils.set_random_seed(42)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Input, Dropout, Dense, RandomFlip,RandomRotation, RandomZoom, Rescaling,GlobalAveragePooling2D)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

BASE = 'datasets/bigdataset/'
print("Looking in:", os.path.abspath(BASE))
print("Contents:", os.listdir(BASE))
INPUT_SIZE = 128
valid_ext = ('.jpg', '.jpeg', '.png')

def load_split(split):
    X, y = [], []
    for cls in sorted(os.listdir(BASE + split)):
        folder = os.path.join(BASE, split, cls)
        if not os.path.isdir(folder):
            continue
        lab = 0 if cls.lower() == 'notumor' else 1   
        for name in os.listdir(folder):
            if name.lower().endswith(valid_ext):
                img = cv2.imread(os.path.join(folder, name))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
                X.append(img)
                y.append(lab)
    return np.array(X, dtype='uint8'), np.array(y)

x_trainval, y_trainval = load_split('Training/')
x_test, y_test = load_split('Testing/')

print("Training folder:", len(x_trainval),
      "| No tumor:", np.sum(y_trainval == 0), "| Tumor:", np.sum(y_trainval == 1))
print("Testing folder:", len(x_test),
      "| No tumor:", np.sum(y_test == 0), "| Tumor:", np.sum(y_test == 1))

# Validation = 15% of the Training folder
x_train, x_val, y_train, y_val = train_test_split(
    x_trainval, y_trainval, test_size=0.15, random_state=0, stratify=y_trainval)
print("Train:", len(x_train), "| Val:", len(x_val), "| Test:", len(x_test))

# Class weights to handle the 3:1 imbalance
w = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
class_weights = dict(enumerate(w))
print("Class weights:", class_weights)

y_train_c = to_categorical(y_train, 2)
y_val_c = to_categorical(y_val, 2)
y_test_c = to_categorical(y_test, 2)

# Transfer learning model (scaling is done inside the model, so images stay uint8)
base = MobileNetV2(input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
                   include_top=False, weights='imagenet')
base.trainable = False

model = Sequential([
    Input(shape=(INPUT_SIZE, INPUT_SIZE, 3)),
    RandomFlip('horizontal'),
    RandomRotation(0.1),
    RandomZoom(0.1),
    Rescaling(1 / 127.5, offset=-1),
    base,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

early = EarlyStopping(monitor='val_loss', patience=6,
                      restore_best_weights=True)

history = model.fit(x_train, y_train_c,
                    class_weight=class_weights,
                    batch_size=32,
                    epochs=30,
                    validation_data=(x_val, y_val_c),
                    callbacks=[early],
                    shuffle=True,
                    verbose=1)

print("\nEpochs actually run:", len(history.history['loss']))
print("Final Training Accuracy:", history.history['accuracy'][-1])
print("Final Validation Accuracy:", history.history['val_accuracy'][-1])

test_loss, test_acc = model.evaluate(x_test, y_test_c, verbose=0)
print("\nTest accuracy:", test_acc)

y_pred = np.argmax(model.predict(x_test), axis=1)
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=['No Tumor', 'Tumor']))

model.save('BrainTumor_big_mobilenet.keras')