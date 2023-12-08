import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import time
from torchvision.transforms import functional as F
from add_noise import AddNoise

def main():
    # Start timing
    start_time = time.time()

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Load and preprocess CIFAR-10 dataset
    transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.ToTensor()])

    transform_with_noise = transforms.Compose(
        [AddNoise(30, True),
         transforms.Resize(256),
         transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=2)

    testset_with_noise = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                      download=True, transform=transform_with_noise)
    testloader_with_noise = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    # Step 2: Define the ResNet50 Model
    net = torchvision.models.resnet50(pretrained=True)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 10) # CIFAR10 has 10 classes

    # Transfer the model to GPU
    net.to(device)

    # Step 3: Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    print('Starting Training')

    # Step 4: Train the Model
    for epoch in range(3): 
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')

    print('Finished Training')

    end_time = time.time()

    duration = end_time - start_time

    print(f"Duration: {duration} seconds")

    # Step 5: Test the Model
    y_pred = []
    y_true = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    print('Finished Testing')

    # Step 6: Calculate and Print Metrics
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Classification Report (Precision, Recall, F1-score)
    class_report = classification_report(y_true, y_pred, target_names=trainset.classes)
    print("Classification Report:")
    print(class_report)

    # Step 7: Test the Model with noise
    y_pred = []
    y_true = []
    with torch.no_grad():
        for data in testloader_with_noise:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    print('Finished Testing')

    # Step 8: Calculate and Print Metrics with noise
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Classification Report (Precision, Recall, F1-score)
    class_report = classification_report(y_true, y_pred, target_names=trainset.classes)
    print("Classification Report:")
    print(class_report)


if __name__ == '__main__':
    main()
