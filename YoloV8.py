import torch
import torchvision
import torchvision.transforms as transforms
from ultralytics import YOLO
import time
from PIL import Image
import os
from sklearn.metrics import confusion_matrix, classification_report
import io
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def test_noise_dataset():
    model = YOLO('Yolov8.pt')  
    trainset_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    noisydata_path = os.path.join("./", "noisydata")
    y_pred = []
    y_true = []
    for class_folder in os.listdir(noisydata_path):
        class_path = os.path.join(noisydata_path, class_folder)

        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)

            with open(image_path, 'rb') as file:
                image = Image.open(io.BytesIO(file.read()))

                result = model(image)
                #print(model.names[result[0].probs.top1])
            y_pred.append(model.names[result[0].probs.top1])
            y_true.append(class_folder)

    # Classification Report (Precision, Recall, F1-score)
    class_report = classification_report(y_true, y_pred, target_names=trainset_classes)
    print("Classification Report:")
    print(class_report)

    return

def main():
    # Start timing
    start_time = time.time()

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load a pre-trained YOLO V8 model
    model  = YOLO('yolov8n-cls.pt')  

    model.to(device)
    results = model.train(data='cifar10', epochs=3, imgsz=32)
    
    end_time = time.time()

    duration = end_time - start_time

    print(f"Duration: {duration} seconds")

    metrics = model.val()  
        

if __name__ == '__main__':
    #main()
    test_noise_dataset()
