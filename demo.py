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

def predict(image, model, selected_model):

    if selected_model == 'YoloV8':
        result = model(image)
        return 'Prediction: ' + model.names[result[0].probs.top1]
    elif selected_model == 'ResNet50' or selected_model == 'AlexNet':
        model.eval()
        
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])

        transformed_image  = transform(image)

        # Make a prediction
        with torch.no_grad():
            prediction = model(transformed_image)

        predicted_class = torch.argmax(prediction, dim=1)
        return predicted_class

# Main Streamlit application
def main():
    st.title("Image Upload and Model Prediction")

    # List of available models
    model_names = ['ResNet50', 'YoloV8', 'AlexNet']  

    # Checkbox for selecting a model
    selected_model = st.selectbox('Select a model for prediction', model_names)

    if selected_model == 'ResNet50':
        model = torchvision.models.resnet50(pretrained=True)
        model.load_state_dict(torch.load('resnet50.pth'))
    elif selected_model == 'YoloV8':
        model = YOLO('./Yolov8.pt')  
    elif selected_model == 'AlexNet':
        model = models.alexnet(pretrained=True)
        model.classifier[6] = nn.Linear(4096, 10)
        model.load_state_dict(torch.load('./alexnet.pth'))

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
            st.write(predict(image, model, selected_model))


if __name__ == '__main__':
    main()
