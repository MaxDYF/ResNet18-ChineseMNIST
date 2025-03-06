import torch

class CommonBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.BatchNorm2d(
                out_channels,
            ),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.BatchNorm2d(
                out_channels
            )
        )
    def forward(self, x):
        initial_x = x
        x = self.layer1(x)
        x = self.layer2(x)
        x += initial_x
        x = torch.nn.functional.relu(x, inplace=True)
        return x

class SpecialBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.change_channel = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=2,
            padding=0
        )
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            torch.nn.BatchNorm2d(
                out_channels,
            ),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.BatchNorm2d(
                out_channels
            )
        )
    def forward(self, x):
        initial_x = self.change_channel(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x += initial_x
        x = torch.nn.functional.relu(x, inplace=True)
        return x


class ResNet18(torch.nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.prepare_layer = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3
            ),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.layer1 = torch.nn.Sequential(
            CommonBlock(64, 64),
            CommonBlock(64, 64)
        )
        self.layer2 = torch.nn.Sequential(
            SpecialBlock(64, 128),
            CommonBlock(128, 128)
        )
        self.layer3 = torch.nn.Sequential(
            SpecialBlock(128, 256),
            CommonBlock(256, 256)
        )
        self.layer4 = torch.nn.Sequential(
            SpecialBlock(256, 512),
            CommonBlock(512, 512)
        )
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc_layer = torch.nn.Sequential(
            torch.nn.Dropout(
                p = 0.5
            ),
            torch.nn.Linear(
                in_features=512,
                out_features=256
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(
                p = 0.5
            ),
            torch.nn.Linear(
                in_features=256,
                out_features=classes
            )
        )
    def forward(self, x):
        x = self.prepare_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc_layer(x)
        return x