import torch
import torch.nn as nn

class fc1(nn.Module):

    def __init__(self, num_classes=10, width=512):
        super(fc1, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(28*28, width),
            # nn.ReLU(inplace=True),
            # nn.Linear(512, 100),
            nn.ReLU(inplace=False),
            nn.Linear(width, num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x