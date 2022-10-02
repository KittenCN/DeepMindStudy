from matplotlib import pyplot as plt
import torch
from torch.nn.modules import module
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def show(image):
    if device.type == "cuda":
        image = image.cpu()
    image = np.array(image, dtype='float')
    pixels = image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()

train_dataset = datasets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='data', train=False, transform=transforms.ToTensor())
train_loader = DataLoader(dataset = train_dataset, batch_size = 512, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size= 512, shuffle = True)

class alexnet(nn.Module):  
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc1 = nn.Linear(256 * 3 * 3, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        out = self.conv1(x) # 28*28*1 -> 5*5*96 -> 2*2*96
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = out.view(out.size(0), -1)

        out = F.relu(self.fc1(out))  # 256*6*6 -> 4096
        out = F.dropout(out, 0.5)
        out = F.relu(self.fc2(out))
        out = F.dropout(out, 0.5)
        out = self.fc3(out)
        out = F.log_softmax(out, dim=1)

        return out

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 224*224*3 -> 224*224*6
        self.maxpool = nn.MaxPool2d(2, 2)  # 224*224*6 -> 112*112*6
        self.conv2 = nn.Conv2d(6, 16, 5)  # 112*112*6 -> 112*112*16
        self.fc1 = nn.Linear(16 * 53 * 53, 1024)  # 16*53*53 -> 1024
        self.fc2 = nn.Linear(1024, 512)  # 1024 -> 512
        self.fc3 = nn.Linear(512, 2)  # 512 -> 2

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)  # 112*112*16 -> 16*53*53
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        # 卷积层
        # i --> input channels
        # 6 --> output channels
        # 5 --> kernel size
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # 全连接层
        # 16 * 4 * 4 --> input vector dimensions
        # 120 --> output vector dimensions
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 卷积 --> ReLu --> 池化
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))

        # reshape, '-1'表示自适应
        # x = (n * 16 * 4 * 4) --> n : input channels
        # x.size()[0] == n --> input channels
        x = x.view(x.size()[0], -1)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class mnist_net(nn.Module):
    def __init__(self) -> None:
        super(mnist_net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1)  # 28 * 28 
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)  # 28 * 28 
        self.pool1 = nn.MaxPool2d(2, 2)  # 14 * 14
        self.fc1 = nn.Linear(128 * 14 * 14, 1024)  
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 10)
        # -----------------------------
        # self.conv1 = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1),
        #                             nn.ReLU(),
        #                             nn.Conv2d(64, 128, 3, 1, 1),
        #                             nn.ReLU(),
        #                             nn.MaxPool2d(2, 2))
        # self.dense = nn.Sequential(nn.Linear(128 * 14 * 14, 1024),
        #                             nn.ReLU(),
        #                             nn.Dropout(p=0.5),
        #                             nn.Linear(1024, 10))     
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = x.view(-1, 128 * 14 * 14)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop1(x)
        x = self.fc2(x)
        # -----------------------------
        # x = self.conv1(x)
        # x = x.view(-1, 128 * 14 * 14)
        # x = self.dense(x)
        return x

# net = mnist_net().to(device)
net = alexnet().to(device)
print(net)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
epochs = 1

for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss_value = loss(outputs, labels)
        loss_value.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f'Epoch: {epoch}, Step: {i}, Loss: {loss_value.item()}')
torch.save(net.state_dict(), "Mnist/model/model.pkl")
# print("------------------------------------------------------")
# correct = 0
# net = torch.load("Mnist/model/model.pkl").to(device)
# print(net)
# for data in test_loader:
#     images, labels = data
#     images = images.to(device)
#     labels = labels.to(device)
#     outputs = net(images)
#     _, predicted = torch.max(outputs.data, 1)
#     correct += (predicted == labels).sum().item()
# print(f'Accuracy: {correct / len(test_loader)}%')
