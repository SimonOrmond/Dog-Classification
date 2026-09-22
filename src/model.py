import torch.nn as nn


class ResidualBlock(nn.Module):
    """Two 3x3 convs, then adds the block's input back onto the output (the "shortcut").

    The shortcut means the block only has to learn what to change about its input,
    and gives gradients a direct path back to early layers, so a deeper network still trains.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # stride=2 halves height and width, doing the job max pooling did before
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        # The input can only be added to the output if their shapes match. When the block
        # changes the channel count or shrinks the image, a 1x1 conv reshapes the input to fit.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)  # the residual connection
        return self.relu(out)


class DogCNN(nn.Module):
    """A small ResNet: a stem conv, four residual blocks (8 convs), global average pooling, one fully connected layer."""

    def __init__(self, num_classes=10):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block1 = ResidualBlock(32, 32, stride=1)
        self.block2 = ResidualBlock(32, 64, stride=2)
        self.block3 = ResidualBlock(64, 128, stride=2)
        self.block4 = ResidualBlock(128, 256, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)     # (batch, 32, 75, 75)
        x = self.block1(x)   # (batch, 32, 75, 75)
        x = self.block2(x)   # (batch, 64, 38, 38)
        x = self.block3(x)   # (batch, 128, 19, 19)
        x = self.block4(x)   # (batch, 256, 10, 10)

        x = self.global_pool(x)
        x = self.flatten(x)  # (batch, 256)
        x = self.fc(x)       # (batch, 10)
        return x
