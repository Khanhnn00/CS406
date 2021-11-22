import torch
import torch.nn as nn
import torch.nn.functional as F

def conv(in_channels, out_channels, k_size, padding=0, stride=1):
    """3x3 kernel size with padding convolutional layer in ResNet BasicBlock."""
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=k_size,
        stride=stride,
        padding=padding,
        bias=False)

class PaperNet(nn.Module):
    """Residual Neural Network."""

    def __init__(self,  num_classes=101):
        """Residual Neural Network Builder."""
        super(PaperNet, self).__init__()

        self.conv1 = conv(3, 30, k_size=6, padding=1, stride=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=False)
        self.conv2 = conv(30, 40, k_size=3, stride=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=False)
        self.conv3 = conv(40, 60, k_size=3, stride=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(in_features=7260, out_features=140, bias=True)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=140, out_features=num_classes, bias=True)
        self.softmax = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        """Forward pass of ResNet."""
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu4(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out
        
