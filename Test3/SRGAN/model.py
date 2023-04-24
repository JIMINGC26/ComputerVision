import torch
import torch.nn as nn
import math

class Block(nn.Module):
    def __init__(self, input_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channels, out_channels, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),

            nn.Conv2d(out_channels, out_channels, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x0):
        x1 = self.layer(x0)
        return x1 + x0

class Generator(nn.Module):
    def __init__(self, scale=1) -> None:
        """放大倍数为scale的平方"""

        upsample_block_num = int(math.log(scale, 2))

        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )
        self.residual_blocks = nn.Sequential(
            Block(),
            Block(),
            Block(),
            Block(),
            Block()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)

        )
        """
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 256, 3, stride=1, padding=1),
            nn.PixelShuffle(scale),
            nn.PReLU(),

            nn.Conv2d(64, 256, 3, stride=1, padding=1),
            nn.PixelShuffle(scale),
            nn.PReLU(),
        )
        """
        conv3 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        self.conv3 = nn.Sequential(*conv3) 
        self.conv4 = nn.Conv2d(64, 3, 9, stride=1, padding=4)

    def forward(self, x):
        x0 = self.conv1(x)
        x = self.residual_blocks(x0)
        x = self.conv2(x)
        x = self.conv3(x + x0)
        x = self.conv4(x)
        return (torch.tanh(x)+1) /2


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class DownSalmpe(nn.Module):
    def __init__(self, input_channel, output_channel,  stride, kernel_size=3, padding=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.down = nn.Sequential(
            DownSalmpe(64, 64, stride=2, padding=1),
            DownSalmpe(64, 128, stride=1, padding=1),
            DownSalmpe(128, 128, stride=2, padding=1),
            DownSalmpe(128, 256, stride=1, padding=1),
            DownSalmpe(256, 256, stride=2, padding=1),
            DownSalmpe(256, 512, stride=1, padding=1),
            DownSalmpe(512, 512, stride=2, padding=1),
        )
        self.dense = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.down(x)
        x = self.dense(x)
        return x


if __name__ == '__main__':
    g = Generator()
    a = torch.rand([1, 3, 64, 64])
    print(g(a).shape)
    d = Discriminator()
    b = torch.rand([2, 3, 512, 512])
    print(d(b).shape)