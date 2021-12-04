import paddle
from paddle.fluid.framework import Parameter
from paddle.vision.transforms import Compose, Normalize
import paddle.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
from paddle.metric import Accuracy

transform = Compose([Normalize(mean = [127.5], 
                        std = [127.5], 
                        data_format = "CHW")])
train_dataset = paddle.vision.datasets.MNIST(mode = "train", transform = transform)
test_dataset = paddle.vision.datasets.MNIST(mode = "test", transform = transform)
print("load finished")

class LeNet(paddle.nn.Layer):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels = 1, out_channels = 6, kernel_size = 5, stride = 1, padding = 2)  # 28*28*6
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size = 2, stride = 2)  # 14*14*6
        self.conv2 = paddle.nn.Conv2D(in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1)  # 10*10*16
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size = 2, stride = 2)  # 5*5*16
        self.linear1 = paddle.nn.Linear(in_features = 5*5*16, out_features = 120)
        self.linear2 = paddle.nn.Linear(in_features = 120, out_features = 84)
        self.linear3 = paddle.nn.Linear(in_features = 84, out_features = 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = paddle.flatten(x, start_axis = 1, stop_asix = -1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x

model = paddle.Model(LeNet())
optim = paddle.optimizer.Adam(learning_rate = 0.001, Parameter = model.parameters())

model.prepare(
    optim,
    paddle.nn.CrossEntropyLoss(),
    Accuracy()
)

model.fit(
    train_dataset,
    batch_size = 64,
    epochs = 2,
    verbose = 1
)