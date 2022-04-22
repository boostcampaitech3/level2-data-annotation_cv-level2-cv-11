import torch
import torch.nn as nn

# Discriminator를 직관적으로 작성한 코드
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=4, stride=2, padding=1, padding_mode="reflect")
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, padding_mode="reflect")
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, padding_mode="reflect")
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, padding_mode="reflect")
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")

        self.relu = nn.LeakyReLU(0.2)
        self.norm2 = nn.InstanceNorm2d(128)
        self.norm3 = nn.InstanceNorm2d(256)
        self.norm4 = nn.InstanceNorm2d(512)

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.relu(self.norm3(self.conv3(x)))
        x = self.relu(self.norm4(self.conv4(x)))
        x = self.conv5(x)
        return x