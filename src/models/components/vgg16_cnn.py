from torch import nn
import torch.nn.functional as F


class VGG16(nn.Module):
    def __init__(
        self,
        input_size: int = 3,
        conv1_size: int = 64,
        conv2_size: int = 128,
        conv3_size: int = 256,
        conv4_size: int = 512,
        lin1_size: int = 4096,
        output_size: int = 1000,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        maxpool_kernel_size: int = 2,
        maxpool_stride: int = 2,
        dropout: float = 0.5
    ):
        super().__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_size,
                      out_channels=conv1_size,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=conv1_size,
                      out_channels=conv1_size,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpool_kernel_size,
                         stride=maxpool_stride)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=conv1_size,
                      out_channels=conv2_size,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=conv2_size,
                      out_channels=conv2_size,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpool_kernel_size,
                         stride=maxpool_stride)
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels=conv2_size,
                      out_channels=conv3_size,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=conv3_size,
                      out_channels=conv3_size,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=conv3_size,
                      out_channels=conv3_size,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpool_kernel_size,
                         stride=maxpool_stride)
        )

        self.block_4 = nn.Sequential(
            nn.Conv2d(in_channels=conv3_size,
                      out_channels=conv4_size,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=conv4_size,
                      out_channels=conv4_size,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=conv4_size,
                      out_channels=conv4_size,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpool_kernel_size,
                         stride=maxpool_stride)
        )

        self.block_5 = nn.Sequential(
            nn.Conv2d(in_channels=conv4_size,
                      out_channels=conv4_size,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=conv4_size,
                      out_channels=conv4_size,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=conv4_size,
                      out_channels=conv4_size,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpool_kernel_size,
                         stride=maxpool_stride)
        )

        self.features = nn.Sequential(
            self.block_1, self.block_2,
            self.block_3, self.block_4,
            self.block_5
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, lin1_size),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(lin1_size, lin1_size),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(lin1_size, output_size),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1000, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        logits = self.fc2(x)
        return F.softmax(logits, dim=1)


if __name__ == "__main__":
    _ = VGG16()
