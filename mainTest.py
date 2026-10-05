import cv2
import numpy as np
from keras.models import load_model

model = load_model('BrainTumor10EpochsCategorical.h5')

image = cv2.imread(
    r'D:\Deep Learning Project\Brain Tumor Image Classification\pred\pred0.jpg'
)

image = cv2.resize(image, (64, 64))

image = image.astype('float32') / 255.0

input_img = np.expand_dims(image, axis=0)

prediction = model.predict(input_img)

print("Prediction probabilities:", prediction)

result = np.argmax(prediction, axis=1)

print("Predicted Class:", result[0])

if result[0] == 0:
    print("No Brain Tumor")
else:
    print("Yes Brain Tumor")