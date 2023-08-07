import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import optim
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import time


font = font_manager.FontProperties(weight='bold', style='normal', size=20)


class VGG(nn.Module):
	def __init__(self, features, num_classes=10, init_weights=True):
		super(VGG, self).__init__()

		self.features = features

		self.classifier = nn.Sequential(nn.Linear(in_features=512 * 3 * 3, out_features=4096),
										nn.ReLU(inplace=True),
										nn.Dropout(p=0.3),

										nn.Linear(in_features=4096, out_features=4096),
										nn.ReLU(inplace=True),
										nn.Dropout(p=0.3),

										nn.Linear(in_features=4096, out_features=num_classes),
										)

		if init_weights:
			self._initialize_weights()

	def forward(self, x):
		x = self.features(x)

		#print(x.shape)
		#sdf

		x = x.view(x.size(0), -1)

		x = self.classifier(x)

		return x

	def _initialize_weights(self):
		for module in self.modules():
			if isinstance(module, nn.Conv2d):
				nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

				if module.bias is not None:
					nn.init.constant_(module.bias, 0)
			elif isinstance(module, nn.BatchNorm2d):
				nn.init.constant_(module.weight, 1)
				nn.init.constant_(module.bias, 0)
			elif isinstance(module, nn.Linear):
				nn.init.normal_(module.weight, 0, 0.01)
				nn.init.constant_(module.bias, 0)


configs = {#'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
		   'A': [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],

		   'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
		   'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
		   'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
		   }

def make_layers(config, batch_norm=True):
	layers = []

	in_channels = 1

	for value in config:
		if value == 'M':
			layers += [nn.MaxPool2d(kernel_size=(2, 2))]
		else:
			conv2d = nn.Conv2d(in_channels=in_channels, out_channels=value, kernel_size=(2, 2), stride=(1, 1), padding='same')

			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(value), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]

			in_channels = value

	return nn.Sequential(*layers)

def vgg11():
	model = VGG(make_layers(configs['A'], batch_norm=True))
	return model

def vgg13():
	model = VGG(make_layers(configs['B'], batch_norm=True))
	return model

def vgg16():
	model = VGG(make_layers(configs['D'], batch_norm=True))
	return model

def vgg19():
	model = VGG(make_layers(configs['E'], batch_norm=True))
	return model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = datasets.MNIST(root='dataset/', train=True, download=True,
							   transform=transforms.Compose([transforms.ToTensor(),
                               			 transforms.Normalize(mean=(0.1307,), std=(0.3081,))
                               			 ])
							   )

val_dataset = datasets.MNIST(root='dataset/', train=False, download=True,
							 transform=transforms.Compose([transforms.ToTensor(),
                               		   transforms.Normalize(mean=(0.1307,), std=(0.3081,))
                               		   ])
							 )

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=True)

model = vgg11().to(device)

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

		#X = X.reshape(X.shape[0], -1)

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

			#X = X.reshape(X.shape[0], -1)

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

		#X = X.reshape(X.shape[0], -1)

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