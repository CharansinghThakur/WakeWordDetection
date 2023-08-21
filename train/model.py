import torch
from torch import nn
from torchsummary import summary

class CNNModel(nn.Module):

    def __init__(self, num_classes, device='cpu'):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features = 6528, out_features = num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        logits = self.linear(x)

        return logits

if __name__=="__main__":
    cnn = CNNModel(num_classes = 2)
    # print(cnn)
    print(summary(cnn, (1, 40, 116)))