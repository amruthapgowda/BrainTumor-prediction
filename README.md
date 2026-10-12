# Brain Tumor Detection from MRI Scans

A deep learning project that classifies brain MRI scans as **tumor** or **no tumor** and reports a confidence score. Includes a Flask web app where you upload a scan and get a prediction.

> This is a student project and a screening aid only. It is not a medical diagnosis.

## Results

| | Value |
|---|---|
| Dataset | [DATASET NAME](ADD-KAGGLE-LINK-HERE), about 7,200 MRI images |
| Classes | No tumor vs tumor (glioma, meningioma and pituitary combined as tumor) |
| Model | MobileNetV2 (pretrained on ImageNet) with a new classification head |
| Input size | 128 x 128 |
| Test set | 1,600 images from the dataset's separate Testing folder, not used for training or tuning |
| **Test accuracy** | **95.1%** |
| Tumor class | Precision 1.00, recall 0.94 |
| No tumor class | Precision 0.84, recall 0.99 |

Confusion matrix on the test set:

| | Predicted no tumor | Predicted tumor |
|---|---|---|
| **Actual no tumor** | 395 | 5 |
| **Actual tumor** | 74 | 1,126 |

## How the project evolved

1. **Version 1 (`mainTrain.py`)**: custom CNN (3 convolution layers, dropout 0.5, softmax output) on a small Kaggle dataset of 253 images at 64 x 64. Training accuracy reached 100% while validation stayed near 80%, which showed overfitting caused by the small dataset.
2. **Improvements**: added augmentation (flip, rotation, zoom), a separate validation and test split, early stopping, and fixed a loader bug that skipped some image files. Accuracy on a 38-image test set stayed between about 63% and 74%, and transfer learning on the small dataset did not change much, so the dataset size was the main limit.
3. **Version 2 (`bigmainTrain.py`)**: moved to a larger public dataset and trained a MobileNetV2 transfer learning model with augmentation, class weights for the 3:1 class imbalance, and early stopping.

## Model details

- Preprocessing: resize to 128 x 128, convert to RGB, pixel rescaling inside the model
- Augmentation (training only): random flip, rotation, zoom
- Base model: MobileNetV2, frozen, followed by global average pooling, dropout 0.3 and a 2-unit softmax layer
- Loss and optimizer: categorical cross-entropy, Adam
- Training: batch size 32, up to 30 epochs, early stopping on validation loss (best weights restored)
- The softmax output is the confidence score shown in the app

## Project files

| File | Purpose |
|---|---|
| `mainTrain.py` | Version 1: custom CNN on the small dataset (`datasets/no`, `datasets/yes`) |
| `bigmainTrain.py` | Version 2: MobileNetV2 on the large dataset (`datasets/bigdataset/Training` and `Testing`) |
| `mainTest.py` | Predicts a single image from the command line |
| `app.py` | Flask web app with a `/predict` endpoint that returns JSON |
| `templates/`, `static/` | Web page for the app |
| `BrainTumor_big_mobilenet.keras` | Trained final model |
| `BrainTumor10EpochsCategorical.h5` | Model from Version 1 |

The datasets are not included in this repository. Download them from the link above.

## Run the app

```
pip install tensorflow flask opencv-python numpy scikit-learn pillow
python app.py
```

Then open http://127.0.0.1:5000 and upload an MRI image.

`/predict` accepts a POST request with an image in the `file` field and returns JSON, for example:

```
{"result": "Tumor Detected", "confidence": 97.31}
```

## Limitations

- Binary classification only: it does not identify the tumor type or grade, or locate the tumor
- It missed 74 of 1,200 tumor scans in the test set (recall 0.94), which matters in a medical setting
- Trained on one public dataset. It has not been checked for duplicate images or split by patient, and it has not been tested on scans from other hospitals or scanners
- Not clinically validated

## Future work

- Tune the decision threshold and fine-tune the base model to raise tumor recall
- K-fold cross-validation for a more stable accuracy estimate
- Four-class classification (glioma, meningioma, pituitary, no tumor)
- Grad-CAM heatmaps and tumor segmentation
- External validation on data from other sources
