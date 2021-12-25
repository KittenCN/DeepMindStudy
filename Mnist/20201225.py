import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

train_dataset = datasets.MNIST(root = 'Mnist/data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='Mnist/data', train=False, transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=True)


class mnist(nn.Module):
    def __init__(self):
        super(mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128*12*12, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x) 
        x = F.relu(x)
        x = self.conv2(x) 
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 128*12*12)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = mnist()
    model = torch.load("abc.pt")
    print(model)
    epochs = 3
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            loss_value = loss(model(images), labels)
            loss_value.backward()
            optimizer.step()
            if i % 10:
                print(loss_value.item())
                torch.save(model.state_dict(), "abc.pt")

