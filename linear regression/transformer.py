import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class TransformerRegression(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_layers):
        super(TransformerRegression, self).__init__()

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers)
        
        self.fc1 = nn.Linear(input_dim, d_model)
        self.fc2 = nn.Linear(d_model, output_dim)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        
        # expand tensor to have shape (sequence length=1, batch size, d_model)
        x = x.unsqueeze(0)
        
        # apply transformer encoder layer
        x = self.transformer_encoder(x)
        
        # squeeze sequence length dimension
        x = x.squeeze(0)
        
        x = self.fc2(x)
        
        return x

# generate random data for y=wx+b prediction
x_train = torch.randn((1000, 1))
y_train = 3 * x_train + 5

# create data loader
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# instantiate model
model = TransformerRegression(input_dim=1, output_dim=1, d_model=512, nhead=8, num_layers=6)

# define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train model
num_epochs = 10
for epoch in range(num_epochs):
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # forward pass
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        
        # backward pass
        loss.backward()
        optimizer.step()
        
        # print training progress
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
