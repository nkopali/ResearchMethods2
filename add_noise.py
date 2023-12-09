#!/usr/bin/env python3

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import time


# Run this file in the same location as your cifar dataset to get a copy with noise added to the test subset
# Alternatively, use this as a pytorch transform

# apply this before any other transform
class AddNoise(torch.nn.Module):
    def __init__(self, amplitude, monochromatic):
        super().__init__()
        self.amplitude = amplitude
        self.monochromatic = monochromatic


    def forward(self, img):
        from PIL import Image, ImageChops
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """

        print(img)
        if self.monochromatic:
            imdata = np.random.randint(0, self.amplitude, (32, 32), dtype=np.uint8)
            noise = Image.fromarray(imdata).convert('RGB')
            ImageChops.add(img, noise).show()
            return ImageChops.add(img, noise);
        else:
            imdata = np.random.randint(0, self.amplitude, (32, 32, 3), dtype=np.uint8)
            noise = Image.fromarray(imdata)
            return ImageChops.add(img, noise);

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{size}"


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def pickle(file, obj):
    import pickle
    with open(file, 'wb') as fo:
        pickle.dump(obj, fo, -1)
    return dict

def save(name, raw):
    import numpy as np
    from PIL import Image

    imdata =  np.random((32, 32, 3), dtype=np.uint8)

    for i in range(32):
        for j in range(32):
            imdata[i][j][0] = raw[i+32*j]
            imdata[i][j][1] = raw[i+32*j + 1024]
            imdata[i][j][2] = raw[i+32*j + 2048]
    # imdata[] = testset['data'][0]
    img = Image.fromarray(imdata, 'RGB');
    img.save(name);

def main():
    import subprocess
    import random

    noise_amp = 30
    monochromatic_noise = True

    print("attempting to unpack tar")
    subprocess.run(["tar", "-xzvf", "cifar-10-python.tar.gz"])

    print("loading test set")
    testset = unpickle("cifar-10-batches-py/test_batch")

    # save('clean.png', testset['data'][10])

    if monochromatic_noise:
        for i in range(len(testset['data'])):
            for j in range(round(len(testset['data'][0]) / 3)):
                x = random.random() * noise_amp;
                testset['data'][i][j] = max(0, min(x + testset['data'][i][j], 255));
                testset['data'][i][j + 1024] = max(0, min(x + testset['data'][i][j + 1024], 255));
                testset['data'][i][j + 2048] = max(0, min(x + testset['data'][i][j + 2048], 255));
    else:
        for i in range(len(testset['data'])):
            for j in range(len(testset['data'][0])):
                x = random.random() * noise_amp;
                testset['data'][i][j] = max(0, min(x + testset['data'][i][j], 255));


    # save('dirty.png', testset['data'][10])

    testset = pickle("cifar-10-batches-py/test_batch", testset)

    subprocess.run(["tar", "-czvf", "cifar-10-python-noisy.tar.gz", "cifar-10-batches-py"])

    print("cleaning up")
    subprocess.run(["rm", "-rf", "cifar-10-batches-py"])

if __name__ == '__main__':
    main()
