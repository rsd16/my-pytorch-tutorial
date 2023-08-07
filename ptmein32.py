import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import optim
import matplotlib.pyplot as plt
import time


class Dense_Mnist(nn.Module):
	def __init__(self, input_size, num_classes):
		super(Dense_Mnist, self).__init__()

		self.fc1 = nn.Linear(in_features=input_size, out_features=32)
		self.bn1 = nn.BatchNorm1d(num_features=32)
		self.relu1 = nn.ReLU()
		self.dropout1 = nn.Dropout(p=0.05)

		self.fc2 = nn.Linear(in_features=32, out_features=64)
		self.bn2 = nn.BatchNorm1d(num_features=64)
		self.relu2 = nn.ReLU()
		self.dropout2 = nn.Dropout(p=0.1)

		self.fc3 = nn.Linear(in_features=64, out_features=128)
		self.bn3 = nn.BatchNorm1d(num_features=128)
		self.relu3 = nn.ReLU()
		self.dropout3 = nn.Dropout(p=0.2)

		self.fc4 = nn.Linear(in_features=128, out_features=256)
		self.bn4 = nn.BatchNorm1d(num_features=256)
		self.relu4 = nn.ReLU()
		self.dropout4 = nn.Dropout(p=0.3)

		self.fc5 = nn.Linear(in_features=256, out_features=num_classes)

	def forward(self, x):
		x = self.fc1(x)
		x = self.bn1(x)
		x = self.relu1(x)
		x = self.dropout1(x)

		x = self.fc2(x)
		x = self.bn2(x)
		x = self.relu2(x)
		x = self.dropout2(x)

		x = self.fc3(x)
		x = self.bn3(x)
		x = self.relu3(x)
		x = self.dropout3(x)

		x = self.fc4(x)
		x = self.bn4(x)
		x = self.relu4(x)
		x = self.dropout4(x)

		x = self.fc5(x)

		return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# mean (sequence) – Sequence of means for each channel.
# std (sequence) – Sequence of standard deviations for each channel.

# the values 0.1307 and 0.3081 used for the Normalize() transformation below are the global mean and standard deviation
# of the MNIST dataset, we'll take them as a given here.

train_dataset = datasets.MNIST(root='dataset/', train=True, download=True,
							   transform=transforms.Compose([transforms.ToTensor(),
                               			 transforms.Normalize(mean=(0.1307,), std=(0.3081,))])
							   )

val_dataset = datasets.MNIST(root='dataset/', train=False, download=True,
							 transform=transforms.Compose([transforms.ToTensor(),
                               		   transforms.Normalize(mean=(0.1307,), std=(0.3081,))])
							 )

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=True)

#examples = enumerate(train_loader)
#batch_idx, (example_data, example_targets) = next(examples)

model = Dense_Mnist(input_size=784, num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=1e-3)

train_acc_progress = []
train_loss_progress = []

val_acc_progress = []
val_loss_progress = []

for epoch in range(10):
	t0_full_epoch = time.time()

	#model.train()

	train_loss = 0.0
	val_loss = 0.0

	num_correct_train = 0
	num_samples_train = 0

	num_correct_val = 0
	num_samples_val = 0

	for batch_index, (X, y) in enumerate(train_loader):	
		X = X.to(device)
		y = y.to(device)

		X = X.reshape(X.shape[0], -1)

		scores = model(X)

		loss = criterion(scores, y)

		optimizer.zero_grad()

		loss.backward()

		optimizer.step()

		train_loss += loss.item()

		_, y_pred = scores.max(1)
		num_correct_train += (y_pred == y).sum()
		num_samples_train += y_pred.size(0)

	time_only_train = time.time() - t0_full_epoch

	t0_only_val = time.time()

	model.eval()

	with torch.no_grad():
		for batch_index, (X, y) in enumerate(val_loader):
			X = X.to(device)
			y = y.to(device)

			X = X.reshape(X.shape[0], -1)

			scores = model(X)

			loss = criterion(scores, y)

			val_loss += loss.item()

			_, y_pred = scores.max(1)
			num_correct_val += (y_pred == y).sum()
			num_samples_val += y_pred.size(0)

	time_only_val = time.time() - t0_only_val

	train_acc = num_correct_train / num_samples_train
	val_acc = num_correct_val / num_samples_val

	train_acc_progress.append(train_acc)
	train_loss_progress.append(train_loss / len(train_loader))

	val_acc_progress.append(val_acc)
	val_loss_progress.append(val_loss / len(val_loader))

	time_full_epoch = time.time() - t0_full_epoch

	print(f'Epoch {epoch + 1}: Training Loss: {train_loss / len(train_loader)} -- Training Accuracy: {train_acc} -- Validation Loss: {val_loss / len(val_loader)} -- Validation Accuracy: {val_acc} -- Epoch Time: {time_full_epoch} seconds')

model.eval()

num_correct_val = 0
num_samples_val = 0

t0_only_val = time.time()

with torch.no_grad():
	for batch_index, (X, y) in enumerate(val_loader):
		X = X.to(device)
		y = y.to(device)

		X = X.reshape(X.shape[0], -1)

		scores = model(X)

		loss = criterion(scores, y)

		_, y_pred = scores.max(1)
		num_correct_val += (y_pred == y).sum()
		num_samples_val += y_pred.size(0)

	val_loss = loss.item()

time_only_val = time.time() - t0_only_val

val_acc = num_correct_val / num_samples_val

print(f'Final Loss: {val_loss} -- Final accuracy: {val_acc} -- Time Taken for Validation: {time_only_val} seconds')

plt.plot(train_acc_progress, label='Training Accuracy', linewidth=3)
plt.plot(val_acc_progress, label='Validation Accuracy', linewidth=3)
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='best', fontsize=14)
plt.show()

plt.plot(train_loss_progress, label='Training Loss', linewidth=3)
plt.plot(val_loss_progress, label='Validation Loss', linewidth=3)
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best', fontsize=14)
plt.show()

#print(val_loss_progress)
#print(train_loss_progress)
