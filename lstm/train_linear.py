import torch
import model
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 100
n_iters = 10000
input_dim = 1
time_step = 1
hidden_dim = 100
layer_dim = 1
output_dim = 1

a = 10
b = 20
num = 1000

x = torch.rand(num).view(-1, 1)  # [ ?, 1]
xx = torch.randn(num).view(-1, 1)
# hot_pixel = num / random.randint(0, 10)
hot_pixel = 0
y = a * x * x + b
yy = a * xx * xx + b
dataset = TensorDataset(x, y)
testdataset = TensorDataset(xx, yy)
data_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(testdataset, shuffle=True, batch_size=batch_size)

num_epochs = int(n_iters / (len(dataset) / batch_size))
# sess = model.LSTMModel(input_dim, hidden_dim, layer_dim, output_dim, batch_size).to(device)
# sess = model.linear_net().to(device)
sess = model.simpleLSTM(input_dim, hidden_dim, layer_dim, output_dim).to(device)
loss_func = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(sess.parameters(), lr=0.1)
optimizer = torch.optim.Adam(sess.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for i, (data, label) in enumerate(data_loader):
        data = data.reshape(-1, time_step, input_dim).to(device)
        data = data.to(device)
        label = label.to(device)
        outputs = sess(data)
        loss = loss_func(outputs, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0: 
        print('Epoch: {}/{}, Step: {}/{}, Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, len(data_loader), loss.item()))

sess.eval()
with torch.no_grad():
    cnt = 0
    total = 0
    for data, label in test_loader:
        data = data.reshape(-1, time_step, input_dim).to(device)
        label = label.to(device)
        outputs = sess(data)
        index = 0
        for item in outputs:
            cnt += 1
            total += abs(item - label[index]) / label[index]
            index += 1
    print("avg error: ", (total / cnt).to("cpu").numpy())