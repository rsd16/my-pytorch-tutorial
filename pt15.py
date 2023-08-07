import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
 

#Declare transform to convert raw data to tensor
transforms = transforms.Compose([
                                 transforms.ToTensor()
])
 
# Loading Data and splitting it into train and validation data
train = datasets.MNIST('', train = True, transform = transforms, download = True)
train, valid = random_split(train, [50000, 10000])
 
# Create Dataloader of the above tensor with batch size = 32
trainloader = DataLoader(train, batch_size=32)
validloader = DataLoader(valid, batch_size=32)
 
# Building Our Mode
class Network(nn.Module):
    # Declaring the Architecture
    def __init__(self):
        super(Network,self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
 
    # Forward Pass
    def forward(self, x):
        x = x.view(x.shape[0], -1)    # Flatten the images
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
 
model = Network()

if torch.cuda.is_available():
    model = model.cuda()
 
# Declaring Criterion and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
 
# Training with Validation
epochs = 5
min_valid_loss = np.inf
 
for e in range(epochs):
    train_loss = 0.0
    for data, labels in trainloader:
        # Transfer Data to GPU if available
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()
         
        # Clear the gradients
        optimizer.zero_grad()

        # Forward Pass
        target = model(data)

        # Find the Loss
        loss = criterion(target,labels)

        # Calculate gradients
        loss.backward()

        # Update Weights
        optimizer.step()

        # Calculate Loss
        train_loss += loss.item()
     
    valid_loss = 0.0
    model.eval() # Optional when not using Model Specific layer
    for data, labels in validloader:
        # Transfer Data to GPU if available
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()
         
        # Forward Pass
        target = model(data)

        # Find the Loss
        loss = criterion(target,labels)
        
        # Calculate Loss
        valid_loss += loss.item()
 
    print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(trainloader)} \t\t Validation Loss: {valid_loss / len(validloader)}')
     
    if min_valid_loss > valid_loss:
        #print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
         
        # Saving State Dict
        #torch.save(model.state_dict(), 'saved_model.pth')