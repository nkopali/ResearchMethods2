import streamlit as st
from PIL import Image
import io
from ultralytics import YOLO


# Main Streamlit application
def main():
    st.title("Image Upload and Model Prediction")

    model = YOLO('yolov8n-cls.pt')  

    # File uploader 
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(io.BytesIO(uploaded_file.read()))
        
        # Display the image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make a prediction
        if st.button('Predict'):
            result = model.predict(image)
            st.write(result)

if __name__ == '__main__':
    main()
