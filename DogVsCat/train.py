import os.path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms, datasets, models
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
train_dataset = datasets.ImageFolder(root='data/DogVsCat/data/cats_and_dogs_small/train', transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.ImageFolder(root='data/DogVsCat/data/cats_and_dogs_small/test', transform=data_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# 创建模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # o = (h + 2p - k) / s + 1
        # o = (h - k) / s + 1
        self.conv1 = nn.Conv2d(3, 6, 5)  
        self.maxpool = nn.MaxPool2d(2, 2) 
        self.conv2 = nn.Conv2d(6, 16, 5)  
        self.fc1 = nn.Linear(16 * 53 * 53, 1024)  
        self.fc2 = nn.Linear(1024, 512)  
        self.fc3 = nn.Linear(512, 2)  

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x))) #224x224x3 -> 220*220*6 -> 110*110*6
        x = self.maxpool(F.relu(self.conv2(x))) #110*110*6 -> 106*106*16 -> 53*53*16
        x = x.view(-1, 16 * 53 * 53) #将x变成一个行向量，其中每一行是一个样本，每一行的大小是16*53*53
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 加载resnet18 模型，
'''
Net (
  (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
  (maxpool): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear (44944 -> 2048)
  (fc2): Linear (2048 -> 512)
  (fc3): Linear (512 -> 2)
)
'''
# net = models.resnet18(pretrained=False)
# num_ftrs = net.fc.in_features
# net.fc = nn.Linear(num_ftrs, 2)  # 更新resnet18模型的fc模型，
net = Net()
if os.path.exists(pklfile):
    net.load_state_dict(torch.load(pklfile))
    net.eval()

if use_gpu:
    net = net.cuda()
print(net)

# 定义loss和optimizer
cirterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 开始训练
#net.train()
for epoch in range(epochs):
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    for i, data in enumerate(train_loader):
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