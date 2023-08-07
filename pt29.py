"""
An implementation of LeNet CNN architecture.
Video explanation: https://youtu.be/fcOW-Zyb5Bo
Got any questions leave a comment on youtube :)
Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
*    2020-04-05 Initial coding
"""


import torch
import torch.nn as nn # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions


class LeNet(nn.Module):
	def __init__(self):
		super(LeNet, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
		self.relu1 = nn.ReLU()
		self.pool1 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

		self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
		self.relu2 = nn.ReLU()
		self.pool2 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

		self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
		self.relu3 = nn.ReLU()

		self.linear1 = nn.Linear(120, 84)
		self.relu4 = nn.ReLU()

		self.linear2 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.pool1(x)

		x = self.conv2(x)
		x = self.relu2(x)
		x = self.pool2(x)

		x = self.conv3(x) # num_examples x 120 x 1 x 1 --> num_examples x 120
		x = self.relu3(x)

		x = x.reshape(x.shape[0], -1)

		x = self.linear1(x)
		x = self.relu4(x)

		x = self.linear2(x)

		return x

def main():
	print('hello world!')

if __name__ == '__main__':
	main()
