import torch
import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self) -> None:
        super(SRCNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, 
                                             padding=4, padding_mode="replicate"),
                                             nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
    