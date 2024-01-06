import streamlit as st
import torch
import torchvision.transforms as transforms
import timm
import cv2
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np

def get_transforms(img):
    transform = transforms.Compose([
        transforms.Resize((58, 58)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = transform(img)
    return img

class CustomResNext(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, 7)  

    def forward(self, x):
        x = self.model(x)
        return x

def render_inference_section(label_map: dict):
    st.title("Emotion Prediction")

    model = CustomResNext("resnet50", pretrained=False)
    device = torch.device('cpu')
    state = torch.load("../inputs/resnet50_fold0_best.pth", map_location=device)
    input_img = st.file_uploader("Please upload an image in order to predict the emotion.", type=["jpg", "jpeg", "png"])
    if input_img is not None:
        st.subheader("Input Image")
        img = Image.open(input_img).convert('RGB')
        st.image(img, caption='Uploaded Image', use_column_width=True)
        img = get_transforms(img)
        model.load_state_dict(state['model'])
        model.eval()
        with torch.no_grad():
            y_preds = model(img.unsqueeze(0)) 
        predictions = y_preds.softmax(1).cpu().numpy()
        label = np.argmax(predictions)
        decoded_label = list(label_map.keys())[list(label_map.values()).index(label)]
        st.subheader("Prediction")
        st.write(f"Emotion: {decoded_label.upper()}")