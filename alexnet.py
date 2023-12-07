import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import time
from sklearn.metrics import confusion_matrix, classification_report

# Start timing
start_time = time.time()

# Step 1: Load and preprocess CIFAR10 data
transform = transforms.Compose([
    transforms.Resize(224), 
    transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

# Step 2: Modify AlexNet for CIFAR10
alexnet = models.alexnet(pretrained=False)
alexnet.classifier[6] = nn.Linear(alexnet.classifier[6].in_features, 10) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = alexnet.to(device)

# Step 3: Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(alexnet.parameters(), lr=0.001, momentum=0.9)

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
