# Chest_XRay_Classifier

This project is a Chest X-Ray Classifer built using Transfer Learning with the **ResNet50** Model in **PyTorch**. The model can predict whether a chest X-ray shows signs of pneumonia or not. The app is deployed on **Streamlit** for an easy-to-use interface. The model was evaluated using **BinaryAccuracy** metric in **PyTorch**.

## Features

* **Image Upload:** Upload a chest X-ray image for classification.
* **Prediction:** The app uses a fine-tuned ResNet50 model for image classification (**Normal** vs **Pneumonia**).
* **Real-time Results:** Get immediate feedback once the image is uploaded.
* **Confidence Score:** Displays the confidence level for the prediction made by the model.

## Demo

Try the app by visiting [app](https://chest-xray-classifier.streamlit.app/).

## Requirements

To run the app locally, some dependencies are needed.
* streamlit
* torch
* torchvision
* pillow
* numpy

Install the dependencies with:
```
pip install -r requirements.txt
```

## Run the app

1. Clone the repository
```
git clone https://github.com/SujayMann/Chest_XRay_Classifier.git
cd Chest_XRay_Classifier
```
2. Install the dependencies
```
pip install -r requirements.txt
```
3. Run the streamlit app
```
streamlit run app.py
```

## How It Works

1. The app uses a **ResNet50** model pretrained on ImageNet, for transfer learning.
2. The model is fine-tuned to classify chest x-ray images into **Normal** and **Pneumonia** categories.
3. Once the image is uploaded, the model processes the image and passes it to the **ResNet50** model.
4. The model outputs a prediction along with the confidence score which is displayed below the image.

## Dataset

This project uses the **Chest X-Ray Images (Pneumonia)** dataset from **Kaggle**. The dataset consists of chest X-ray images classified as Normal and Pneumonia.
* Dataset link: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
