import torch 
import numpy as np
import torch.nn as nn
import torchvision.models as models
from resnet import resnet18
import torch.nn.functional as F

class Road_map(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = resnet18()
        num_classes = 1
        self.classifier = nn.Conv2d(512, num_classes, kernel_size=3, padding=1, bias=False)
        self.input_shape = (800,800)

    def forward(self, x):
        #print(x.shape)
        x = x.view(self.batch_size, -1, 256, 306).to(self.device)
        x = self.encoder(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=self.input_shape, mode='bilinear', align_corners=False)
        #print(x.shape)
        return x