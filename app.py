import streamlit as st
import torch
from torch import nn
from torchvision.models import resnet50
from torchvision import transforms, datasets
from PIL import Image

torch.manual_seed(42)

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.resnet(x)

model = CustomModel()
state_dict = torch.load('model_resnet50.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=False)

labels = ['Normal', 'Pneumonia']

st.title('Chest X-Ray Classifier')
st.write('Pneumonia Detection from Chest X-rays')

def preprocess_image(image):
    image_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return image_transform(image).unsqueeze(0)

def main():
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

if __name__ == '__main__':
    main()
