import streamlit as st
import torch
from torch import nn
from torchvision.models import resnet50
from torchvision import transforms, datasets
from PIL import Image

model = torch.load('model_resnet50.pth', map_location=torch.device('cpu'))

labels = ['Normal', 'Pneumonia']

def info_page():
    st.title("How to Use the App")
    st.write("""
        This app helps you upload chest x-rays and classify between `NORMAL` and `PNEUMONIA` using a pre-trained CNN model.
        
        **Steps**:
        1. Navigate to the **Predict** page via the sidebar.
        2. Upload an image using the **Browse files** button.
        3. The model will predict if the x-ray shows pneumonia and display the result.
    """)

def preprocess_image(image):
    image_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return image_transform(image).unsqueeze(0)

def prediction_page():

    st.header('Image Upload and Prediction')

    img_path = st.file_uploader(label='Chest X-Ray', type=['png', 'jpg', 'jpeg'])
    if img_path is not None:
        st.image(img_path, caption='Uploaded Image')
        image = Image.open(img_path).convert('RGB')
        image_tensor = preprocess_image(image)
        with torch.no_grad():
            model.eval()
            result = model(image_tensor)
            pred = (result > 0.5).int()
            if pred == 1:
                proba = result.item() * 100
            else:
                proba = (1 - result.item()) * 100

        st.write(f"Prediction: {labels[pred]} ({proba:.2f}%)")

def main():
    pg = st.navigation([
        st.Page(info_page, title="Home", icon=':material/home:'),
        st.Page(prediction_page, title="Predict", icon=':material/stethoscope:'),
    ])

    pg.run()

if __name__ == '__main__':
    main()
