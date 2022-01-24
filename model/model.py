from abc import abstractmethod

import numpy as np
import torch.nn as nn
import torchvision


class BaseModel(nn.Module):
    @abstractmethod
    def forward(self, *inputs):
        raise NotImplementedError

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)


class Densenet201(BaseModel):
    def __init__(self, num_classes):
        super(Densenet201, self).__init__()
        self.model = torchvision.models.densenet201(pretrained=False)
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier.in_features, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )
        # weight initialization
        for m in self.modules:
            if isinstance(m, nn.Linear):
                # using the range [−𝑦,𝑦] , where  𝑦=1/√𝑛 , 𝑛 is the number of inputs to a given neuron.
                # get the number of the inputs
                n = m.in_features
                y = 1.0 / np.sqrt(n)
                m.weight.data.uniform_(-y, y)
                m.bias.data.fill_(0)

    def forward(self, x):
        return self.model(x)


class Densenet121(BaseModel):
    def __init__(self, num_classes):
        super(Densenet201, self).__init__()
        self.model = torchvision.models.densenet121(pretrained=False)
        for param in self.model.parameters():
            param.requires_grad = False
        # new definations have requires grad by default
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier.in_features, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

        # weight initialization
        for m in self.modules:
            if isinstance(m, nn.Linear):
                # using the range [−𝑦,𝑦] , where  𝑦=1/√𝑛 , 𝑛 is the number of inputs to a given neuron.
                # get the number of the inputs
                n = m.in_features
                y = 1.0 / np.sqrt(n)
                m.weight.data.uniform_(-y, y)
                m.bias.data.fill_(0)

    def forward(self, x):
        return self.model(x)
