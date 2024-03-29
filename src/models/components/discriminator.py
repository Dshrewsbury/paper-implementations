from torch import nn


class Discriminator(nn.Module):
    def __init__(self,
                 channels: int = 1,
                 height: int = 28,
                 width: int = 28
                 ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(channels * height * width, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity
