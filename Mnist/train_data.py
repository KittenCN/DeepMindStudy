import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = datasets.MNIST(root='Mnist\data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='Mnist\data', train=False, transform=transforms.ToTensor())
train_loader = DataLoader(dataset = train_dataset, batch_size = 100, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size= 100, shuffle = True)

class mnist_net(nn.Module):
    def __init__(self) -> None:
        super(mnist_net, self).__init__()
        # self.conv1 = nn.Conv2d(1, 64, 3, 1, 1)  # 28 * 28 
        # self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)  # 28 * 28 
        # self.pool1 = nn.MaxPool2d(2, 2)  # 14 * 14
        # self.fc1 = nn.Linear(128 * 14 * 14, 1024)  
        # self.drop1 = nn.Dropout(p=0.5)
        # self.fc2 = nn.Linear(1024, 10)
        # -----------------------------
        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 128, 3, 1, 1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2, 2))
        self.dense = nn.Sequential(nn.Linear(128 * 14 * 14, 1024),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(1024, 10))
    
    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = self.pool1(F.relu(self.conv2(x)))
        # x = x.view(-1, 128 * 14 * 14)
        # x = F.relu(self.fc1(x))
        # x = self.drop1(x)
        # x = self.fc2(x)
        # -----------------------------
        x = self.conv1(x)
        x = x.view(-1, 128 * 14 * 14)
        x = self.dense(x)
        return x

net = mnist_net().to(device)
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

print("------------------------------------------------------")
correct = 0
for data in test_loader:
    images, labels = data
    images = images.to(device)
    labels = labels.to(device)
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    correct += (predicted == labels).sum().item()
print(f'Accuracy: {correct / len(test_loader)}%')
