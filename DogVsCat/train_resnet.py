import os.path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms, datasets, models
from model import resnet34
import matplotlib.pyplot as plt
# from tqdm import trange

# 配置参数
pklfile = 'DogVsCat/model/model.pkl'
random_state = 1
torch.manual_seed(random_state)  # 设置随机数种子，确保结果可重复
torch.cuda.manual_seed(random_state)
torch.cuda.manual_seed_all(random_state)
np.random.seed(random_state)
# random.seed(random_state)
use_gpu = torch.cuda.is_available()
epochs = 10  # 训练次数
batch_size = 16  # 批处理大小

# 对加载的图像作归一化处理， 并裁剪为[224x224x3]大小的图像
data_transform = transforms.Compose([  # 将transforms作为一个整体来使用
    transforms.Scale(256),  # 将图像缩放到最小边为256
    transforms.CenterCrop(224),  # 将图像中心裁剪为224x224 (h, w)或(size, size)
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 将图像归一化  channel=（channel-mean）/std
])

# 数据的批处理，尺寸大小为batch_size,
# 在训练集中，shuffle 必须设置为True, 表示次序是随机的
train_dataset = datasets.ImageFolder(root=r'D:\workstation\GitHub\DeepMindStudy\data\DogVsCat\data\cats_and_dogs_small\train', transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.ImageFolder(root=r'D:\workstation\GitHub\DeepMindStudy\data\DogVsCat\data\cats_and_dogs_small\test', transform=data_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

net = resnet34(5)
if os.path.exists(pklfile):
    net.load_state_dict(torch.load(pklfile))
    net.eval()

if use_gpu:
    net = net.cuda()
print(net)

# 定义loss和optimizer
cirterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

index = 0
x_data = []
y_data = []

# 开始训练
#net.train()
for epoch in range(epochs):
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    for i, data in enumerate(train_loader):
        index += 1
        x_data.append(index)
        inputs, train_labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(train_labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(train_labels)
        # inputs, labels = Variable(inputs), Variable(train_labels)
        optimizer.zero_grad()
        outputs = net(inputs)
        _, train_predicted = torch.max(outputs.data, 1)
        # import pdb
        # pdb.set_trace()
        train_correct += (train_predicted == labels.data).sum()
        loss = cirterion(outputs, labels)
        y_data.append(loss)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_total += train_labels.size(0)
        
        if i % 10 == 0:
            print('train epoch %d id %d epoch loss: %.3f  acc: %.3f ' % (epoch, i, running_loss / train_total, 100 * train_correct / train_total))
        if i % 100 == 0:
            torch.save(net.state_dict(), pklfile)

    # 模型测试
    correct = 0
    test_loss = 0.0
    test_total = 0
    # net.eval()
    for data in test_loader:
        images, labels = data
        if use_gpu:
            images, labels = Variable(images.cuda()), Variable(labels.cuda())
        else:
            images, labels = Variable(images), Variable(labels)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        loss = cirterion(outputs, labels)
        test_loss += loss.data[0]
        test_total += labels.size(0)
        correct += (predicted == labels.data).sum()

    print('test  %d epoch loss: %.3f  acc: %.3f ' % (epoch, test_loss / test_total, 100 * correct / test_total))

    plt.plot(x_data, y_data, 'r')
    plt.show