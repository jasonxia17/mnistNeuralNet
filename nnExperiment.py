import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
np.set_printoptions(suppress=True, precision=3)

from mnist import trainloader, testloader, showImgBatch

class DigitNet(nn.Module):

    def __init__(self):
        super(DigitNet, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 16)
        self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16, 10)

    def forward(self, x):
        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))
        x = self.layer3(x)
        return x

net = DigitNet()

optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for i, batch in enumerate(trainloader):
        imgBatch, labelBatch = batch
        inMatrix = torch.flatten(imgBatch[:,0], start_dim=1)
        optimizer.zero_grad()
        output = net(inMatrix)
        loss = criterion(output, labelBatch)
        loss.backward()
        optimizer.step()

        print(epoch, i)
        if (i + 1) % 1000 == 0:
            print('\n----------------CHECKPOINT--------------', epoch, i)
            
correctCount = 0
wrongCount = 0

for imgBatch, labelBatch in testloader:
    inMatrix = torch.flatten(imgBatch[:,0], start_dim=1)
    output = net(inMatrix)
    loss = criterion(output, labelBatch)

    prediction = F.softmax(output, dim=-1).detach()
    maxProbs, maxes = torch.max(prediction, 1)
    correctProbs = torch.tensor([prediction[i][labelBatch[i]] for i in range(len(prediction))])

    print('\n----------------------------------------------')
    print(prediction.numpy())
    print()

    print('prob for correct label: \t', correctProbs.numpy())
    print('prob for predicted label: \t', maxProbs.numpy())

    print()
    print('correct labels: \t', labelBatch)
    print('predicted labels: \t', maxes)

    print()
    print('loss: ', loss)
    # showImgBatch(imgBatch)

    if int(labelBatch) == int(maxes):
        correctCount += 1
    else:
        wrongCount += 1

print(correctCount, wrongCount)

# stats: 9479 521
# with momentum: 9442 558
