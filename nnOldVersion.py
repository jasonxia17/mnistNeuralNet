import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
np.set_printoptions(suppress=True, precision=3)


class DigitNet(nn.Module):

    def __init__(self):
        super(DigitNet, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 16)
        self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16, 10)

    def forward(self, x):
        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=-1)
        return x


net = DigitNet()
numpyLayer = net.layer3.weight.data.numpy()

twoImages = torch.cat((torch.randn(1, 784), torch.randn(1, 784)))
print(twoImages)

target = torch.zeros(2, 10)
target[0][0] = target[1][3] = 1
print(target)

criterion = nn.MSELoss()
print()

for epoch in range(10001):
    net.zero_grad()
    # This resets the gradients to zero
    output = net(twoImages)
    loss = criterion(output, target)
    loss.backward()

    # loss.backward(retain_graph=True)

    learning_rate = 0.01
    for f in net.parameters():
        f.data.sub_(f.grad.data * learning_rate)

    if epoch % 100 == 0:
        print('\n---------------- EPOCH', epoch, '--------------')
        print(output.detach().numpy())
        print(loss)

print()
apple = torch.tensor([[0.0, 0, 0], [1, 1, 1]])
banana = torch.tensor([[2.0, 0, 1], [1, 5, 1]])
citrus = torch.tensor([2, 1])

print(nn.CrossEntropyLoss()(banana, citrus))

# torch.save(net.state_dict(), 'savemodel.txt')
