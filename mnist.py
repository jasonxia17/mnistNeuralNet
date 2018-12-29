import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import itertools


trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                    transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True)


testset = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                    transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
plt.style.use('grayscale')

def showImgBatch(imgBatch):
    plt.imshow(torch.cat([img[0] for img in imgBatch]))
    plt.show()

if __name__ == '__main__':
    print(len(testset))
    print(len(testloader))

    for imgBatch, labelBatch in testloader:
        print(labelBatch)
        inMatrix = torch.flatten(imgBatch[:,0], start_dim=1)
        print(inMatrix)
        showImgBatch(imgBatch)
        break

    for i in range(150,150):
        img, label = testset[i]
        print(int(label))
        plt.imshow(img.numpy()[0])
        plt.show()

    print(len(trainset))
    print(len(trainloader))