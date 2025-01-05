import streamlit as st
import torch
from torch import nn
from torchvision.models import resnet50
from torchvision import transforms, datasets
from PIL import Image

def main():
    pg = st.navigation([
        st.Page('info.py', title="Home", icon=':material/home:'),
        st.Page('prediction.py', title="Predict", icon=':material/stethoscope:'),
    ])
    pg.run()

if __name__ == '__main__':
    main()