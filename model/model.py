import numpy as np
import torch.nn as nn
import torchvision


class Resnet101(nn.Module):
    def __init__(self, num_classes):
        super(Resnet101, self).__init__()
        self.model = torchvision.models.resnet101(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False

        # new definations have requires grad by default
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )
        for m in self.model.fc:
            if isinstance(m, nn.Linear):
                # using the range [−𝑦,𝑦] , where  𝑦=1/√𝑛 , 𝑛 is the number of inputs to a given neuron.
                # get the number of the inputs
                n = m.in_features
                y = 1.0 / np.sqrt(n)
                m.weight.data.uniform_(-y, y)
                m.bias.data.fill_(0)

    def forward(self, x):
        return self.model(x)


class Resnet50(nn.Module):
    def __init__(self, num_classes):
        super(Resnet50, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False

        # new definations have requires grad by default
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )
        # for m in self.model.fc:
        #     if isinstance(m, nn.Linear):
        #         # using the range [−𝑦,𝑦] , where  𝑦=1/√𝑛 , 𝑛 is the number of inputs to a given neuron.
        #         # get the number of the inputs
        #         n = m.in_features
        #         y = 1.0 / np.sqrt(n)
        #         m.weight.data.uniform_(-y, y)
        #         m.bias.data.fill_(0)

    def forward(self, x):
        return self.model(x)
