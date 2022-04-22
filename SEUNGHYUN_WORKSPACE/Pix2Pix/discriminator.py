import torch
import torch.nn as nn
'''
Instance Norm - https://stackoverflow.com/questions/45463778/instance-normalisation-vs-batch-normalisation
'''
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=padding, bias=False, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels, affine=True), # BatchNorm -> InstanceNorm2d: Artifacts를 없애주는 역할
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]): # 4개의 CNNBlock
        super().__init__()

        # Conv2d + ReLU(No InstanceNorm.)
        self.initial = nn.Sequential(
            # X, y가 concat되어 입력되기 때문에 in_channels x 2만큼을 입력으로 받는다. 
            nn.Conv2d(in_channels*2, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                # 마지막 레이어를 제외하고 stride=2
                CNNBlock(in_channels, feature, stride = 1 if feature == features[-1] else 2),
            )
            in_channels = feature
        
        layers.append(
            nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")
        )

        self.model = nn.Sequential(*layers)
    
    # x,y를 입력으로 받는다. 
    # y: 가짜 이미지 혹은 Ground Truth
    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        return self.model(x)


def test():
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    model = Discriminator()
    preds = model(x, y)
    print(preds.shape)

if __name__ == '__main__':
    test()