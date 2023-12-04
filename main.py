import tensorflow as tf
import torch
import cv2
from torchvision import models, transforms
from PIL import Image
import requests
import deeplake

# Load Dataset
dsTrain = deeplake.load("hub://activeloop/coco-text-train")
dsTest = deeplake.load("hub://activeloop/coco-text-test")

# Load TensorFlow model (e.g., MobileNet)
tf_model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Load PyTorch model (e.g., ResNet)
pytorch_model = models.resnet50(pretrained=True)
pytorch_model.eval()

# For OpenCV/YOLO, you would load the weights and config files
# For Google Cloud Vision, you would set up the client

def preprocess_image_tf(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

def preprocess_image_pytorch(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path)
    img_t = preprocess(img)
    return torch.unsqueeze(img_t, 0)

def run_tf_model(image_path):
    img = preprocess_image_tf(image_path)
    preds = tf_model.predict(img)
    return tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=3)[0]

def run_pytorch_model(image_path):
    img = preprocess_image_pytorch(image_path)
    with torch.no_grad():
        preds = pytorch_model(img)
    return torch.nn.functional.softmax(preds, dim=1).topk(3)
