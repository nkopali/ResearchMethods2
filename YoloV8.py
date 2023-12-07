import torch
import torchvision
import torchvision.transforms as transforms
from ultralytics import YOLO
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

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
    main()
