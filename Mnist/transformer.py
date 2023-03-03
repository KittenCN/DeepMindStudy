import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision import datasets, transforms

# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=28*28, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.decoder_layer = nn.Linear(in_features=28*28, out_features=10)

    def forward(self, x):
        x = x.view(-1, 28*28).T # Transpose image tensor to use it as sequence data
        x = self.transformer_encoder(x)
        x = self.decoder_layer(x[-1])
        return x

# Load the data
# dataset = MNIST(root='data/', download=True, transform=ToTensor())
# train_ds, val_ds = random_split(dataset, [50000, 10000]) # Split into training and validation sets
# train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
# val_loader = DataLoader(val_ds, batch_size=128)

train_dataset = datasets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='data', train=False, transform=transforms.ToTensor())
train_loader = DataLoader(dataset = train_dataset, batch_size = 128, shuffle = True)
val_loader = DataLoader(dataset = test_dataset, batch_size= 128)

# Initialize the model and optimizer
model = TransformerModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train the model
def train(model, dataloader, optimizer):
    model.train() # Set model to training mode
    training_loss = 0.0
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        training_loss += loss.item() * data.size(0)
    training_loss /= len(dataloader.dataset)
    return training_loss

# Evaluate the model
def evaluate(model, dataloader):
    model.eval() # Set model to evaluation mode
    validation_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)
            validation_loss += nn.functional.cross_entropy(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True) # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    validation_loss /= len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)
    return validation_loss, accuracy

for epoch in range(10):
    train_loss = train(model, train_loader, optimizer)
    val_loss, val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch}: \t Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
