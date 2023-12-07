import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

def main():
    # Load and preprocess CIFAR-10 dataset
    transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Load a pre-trained ResNet50 model
    net = torchvision.models.resnet50(pretrained=True)
    net.fc = nn.Linear(net.fc.in_features, 10)  # Adapt the final layer for CIFAR-10

    # Switch the model to evaluation mode
    net.eval()

    # Run inference on some test images
    counter = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            counter += 1

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if counter == 500:  # Just to show a few predictions
                print("#############", correct, total)
                break

if __name__ == '__main__':
    main()
