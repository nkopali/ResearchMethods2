import streamlit as st
from PIL import Image
import io
from ultralytics import YOLO
import torch
import torch.nn as nn
import os
import torchvision
from torchvision import models
import torchvision.transforms as transforms


# Main Streamlit application
def main():
    st.title("Image Upload and Model Prediction")

    model = YOLO('./Yolov8.pt')  

    # File uploader 
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(io.BytesIO(uploaded_file.read()))
        w, h = image.size
        ratio = 32 / min(w, h)
        image = image.resize((round(w * ratio), round(h*ratio)))
        w, h = image.size
        image = image.crop((w/2 - 16, h/2-16, w/2 + 16, h/2+16))
        # Display the image
        st.image(image, caption='Uploaded Image', width=300)

        # Make a prediction
        if st.button('Predict'):
            result = model(image)
            st.write('Prediction: ' + model.names[result[0].probs.top1])


if __name__ == '__main__':
    main()
