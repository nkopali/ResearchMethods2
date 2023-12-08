import streamlit as st
from PIL import Image
import io
from ultralytics import YOLO


# Main Streamlit application
def main():
    st.title("Image Upload and Model Prediction")

    model = YOLO('runs/classify/train/weights/best.pt')  

    # File uploader 
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(io.BytesIO(uploaded_file.read()))
        
        # Display the image
        st.image(image, caption='Uploaded Image', width=300)

        # Make a prediction
        if st.button('Predict'):
            result = model(image)
            st.write(model.names[result[0].probs.top1])

if __name__ == '__main__':
    main()
