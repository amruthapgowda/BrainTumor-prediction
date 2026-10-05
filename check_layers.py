from keras.models import load_model

model = load_model('BrainTumor10EpochsCategorical.h5')

for layer in model.layers:
    print(layer.name)