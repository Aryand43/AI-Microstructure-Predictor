import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from model import MicrostructureCNN
from dataset import MicrostructureDataset

device = torch.device("cpu")

dataset = MicrostructureDataset(
    label_csv='../data/labels.csv',
    image_dir='../data/images'
)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

model = MicrostructureCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for i, (inputs, targets) in enumerate(dataloader):
    print("INPUT:", inputs.shape, inputs.min().item(), inputs.max().item())
    print("TARGET:", targets)
    break

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {running_loss:.4f}")

torch.save(model.state_dict(), 'microstructure_model.pth')
print("Model saved to microstructure_model.pth")
