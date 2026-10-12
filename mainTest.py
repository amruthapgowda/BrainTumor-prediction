import cv2
import numpy as np
from keras.models import load_model

model = load_model('BrainTumor_big_mobilenet.keras')
INPUT_SIZE = 128

image = cv2.imread(r'C:\Users\Admin\Desktop\projects\BrainTumorAI\datasets\bigdataset\Testing\glioma\<that file name>')
if image is None:
    raise SystemExit("Could not read the image. Check the path.")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)     # model was trained on RGB
image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
image = image.astype('float32')                    # no /255: the model rescales inside

input_img = np.expand_dims(image, axis=0)

prediction = model.predict(input_img, verbose=0)

print("Prediction probabilities:", prediction)

result = int(np.argmax(prediction, axis=1)[0])
confidence = float(np.max(prediction) * 100)

print("Predicted Class:", result)

if result == 0:
    print(f"No Brain Tumor ({confidence:.2f}% confidence)")
else:
    print(f"Yes Brain Tumor ({confidence:.2f}% confidence)")