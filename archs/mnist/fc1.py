import torch
import torch.nn as nn

class fc1(nn.Module):

    def __init__(self, num_classes=10, width=512, depth=1.0, bias=True):
        super(fc1, self).__init__()
        standard_depth = 2
        depth = int(standard_depth * depth)
        assert depth >= 1, (standard_depth, depth)
        if depth == 1:
            self.classifier = nn.Linear(28*28, num_classes, bias=bias)
        else:
            self.classifier = nn.Sequential(*([nn.Sequential(
                nn.Linear(28*28, width, bias=bias), 
                nn.ReLU()
            )] + [nn.Sequential(
                nn.Linear(width, width, bias=bias), 
                nn.ReLU()
            ) for _ in range(depth-2)] + [
                nn.Linear(width, num_classes, bias=bias)
            ]))


    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x