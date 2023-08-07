import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import optim
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


font = font_manager.FontProperties(weight='bold', style='normal', size=20)


class Conv_Mnist(nn.Module):
	def __init__(self, in_channels, num_classes):
		super(Conv_Mnist, self).__init__()

		self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(2, 2), stride=(1, 1), padding='same')
		self.bn1 = nn.BatchNorm2d(16)
		self.relu1 = nn.ReLU()
		self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
		self.dropout1 = nn.Dropout(p=0.05)

		self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2), stride=(1, 1), padding='same')
		self.bn2 = nn.BatchNorm2d(32)
		self.relu2 = nn.ReLU()
		self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
		self.dropout2 = nn.Dropout(p=0.1)

		self.conv_skip1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding='same')
		self.bn_skip1 = nn.BatchNorm2d(64)

		self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(1, 1), padding='same')
		self.bn3 = nn.BatchNorm2d(64)
		self.relu3 = nn.ReLU()
		self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
		self.dropout3 = nn.Dropout(p=0.15)

		self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(1, 1), padding='same')
		self.bn4 = nn.BatchNorm2d(128)
		self.relu4 = nn.ReLU()
		self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))
		self.dropout4 = nn.Dropout(p=0.2)

		self.fc1 = nn.Linear(128 * 7 * 7, 256)
		self.bn5 = nn.BatchNorm1d(256)
		self.relu5 = nn.ReLU()
		self.dropout5 = nn.Dropout(p=0.3)

		self.fc2 = nn.Linear(256, 64)
		self.bn6 = nn.BatchNorm1d(64)
		self.relu6 = nn.ReLU()
		self.dropout6 = nn.Dropout(p=0.4)

		self.fc3 = nn.Linear(64, num_classes)

	def forward(self, x):
		source = x

		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu1(x)
		#x = self.pool1(x)
		x = self.dropout1(x)

		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu2(x)
		#x = self.pool2(x)
		x = self.dropout2(x)

		source = self.conv_skip1(source)
		source = self.bn_skip1(source)

		x = self.conv3(x)
		x = self.bn3(x)

		x += source

		x = self.relu3(x)
		x = self.pool3(x)
		x = self.dropout3(x)

		x = self.conv4(x)
		x = self.bn4(x)
		x = self.relu4(x)
		x = self.pool4(x)
		x = self.dropout4(x)

		x = x.reshape(x.shape[0], -1)

		x = self.fc1(x)
		x = self.bn5(x)
		x = self.relu5(x)
		x = self.dropout5(x)

		x = self.fc2(x)
		x = self.bn6(x)
		x = self.relu6(x)
		x = self.dropout6(x)

		x = self.fc3(x)

		return x


'''
# test run to determine the size of fc1 input neurons.
'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# do better transforms
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
val_dataset = datasets.MNIST(root='dataset', train=True, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
val_loader= DataLoader(dataset=val_dataset, batch_size=32, shuffle=True)

model = Conv_Mnist(in_channels=1, num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train_acc_progress = []
train_loss_progress = []

val_acc_progress = []
val_loss_progress = []

for epoch in range(10):
	model.train()

	train_loss = 0.0
	val_loss = 0.0

	num_correct_train = 0
	num_samples_train = 0

	num_correct_val = 0
	num_samples_val = 0

	for batch_index, (X, y) in enumerate(train_loader):
		X = X.to(device)
		y = y.to(device)

		scores = model(X)

		loss = criterion(scores, y)

		optimizer.zero_grad()

		loss.backward()

		optimizer.step()

		train_loss += loss.item()

		_, y_pred = scores.max(1)
		num_correct_train += (y_pred == y).sum()
		num_samples_train += y_pred.size(0)

	model.eval()

	with torch.no_grad():
		for batch_index, (X, y) in enumerate(val_loader):
			X = X.to(device)
			y = y.to(device)

			scores = model(X)

			loss = criterion(scores, y)

			val_loss += loss.item()

			_, y_pred = scores.max(1)
			num_correct_val += (y_pred == y).sum()
			num_samples_val += y_pred.size(0)

	train_acc = num_correct_train / num_samples_train
	val_acc = num_correct_val / num_samples_val

	print(f'Epoch {epoch + 1}: Training Loss: {train_loss / len(train_loader)} -- Training Accuracy: {train_acc} -- Validation Loss: {val_loss / len(val_loader)} -- Validation Accuracy: {val_acc}')

	train_acc_progress.append(train_acc)
	train_loss_progress.append(train_loss / len(train_loader))

	val_acc_progress.append(val_acc)
	val_loss_progress.append(val_loss / len(val_loader))

model.eval()

num_correct_val = 0
num_samples_val = 0

with torch.no_grad():
	for batch_index, (X, y) in enumerate(val_loader):
		X = X.to(device)
		y = y.to(device)

		scores = model(X)

		loss = criterion(scores, y)

		_, y_pred = scores.max(1)
		num_correct_val += (y_pred == y).sum()
		num_samples_val += y_pred.size(0)

	val_loss = loss.item()

val_acc = num_correct_val / num_samples_val

print(f'Final Loss: {val_loss} -- Final accuracy: {val_acc}')

train_acc_progress = [tensor.detach().cpu() for tensor in train_acc_progress]
val_acc_progress = [tensor.detach().cpu() for tensor in val_acc_progress]

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(train_acc_progress, label='Training Accuracy', linewidth=5)
plt.plot(val_acc_progress, label='Validation Accuracy', linewidth=5)
plt.title('Model Accuracy')
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.legend(loc='best', prop=font)
plt.show()

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(train_loss_progress, label='Training Loss', linewidth=5)
plt.plot(val_loss_progress, label='Validation Loss', linewidth=5)
plt.title('Model Loss')
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.legend(loc='best', prop=font)
plt.show()

#print(val_loss_progress)
#print(train_loss_progress)
