import torch
import torch.utils.data
import torch.nn.functional as F

use_gpu = torch.cuda.is_available()

class linear_net(torch.nn.Module):
    def __init__(self):
        super(linear_net, self).__init__()
        # self.linear = torch.nn.Linear(in_features=1, out_features=1)
        self.fc1 = torch.nn.Linear(in_features=1, out_features=8)
        self.fc2 = torch.nn.Linear(in_features=8, out_features=1)
    def forward(self, x):
        # return self.linear(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    x = float(input("Please input x: "))
    model = torch.load("linear regression\model\model.pkl")
    x = torch.tensor([[x]], dtype=torch.float32)
    if use_gpu:
        x = x.cuda()
        model = model.cuda()
    y = model(x)
    print("y = ", y.item())