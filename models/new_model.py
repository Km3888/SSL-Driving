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
        x = self.encoder(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=self.input_shape, mode='bilinear', align_corners=False)
        #print(x.shape)
        return x


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))
        return self.features


class NewRoad_map(nn.Module):

    def __init__(self):
        super(NewRoad_map, self).__init__()
        self.encoder = ResnetEncoder(num_layers=18, pretrained=False)
        self.conv1 = nn.Conv2d(512, 32, 1)
        # self.convTr1 = nn.ConvTranspose2d(192, 128, 10)
        # self.bn1 = nn.BatchNorm2d(128)
        # self.convTr2 = nn.ConvTranspose2d(128, 64, 10)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.convTr3 = nn.ConvTranspose2d(64, 2, 10)
        self.upsample_cls = nn.Sequential(
            nn.ConvTranspose2d(192, 128, 10),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 10),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 2, 10),
        )
    
    def forward(self, samples):
        last_rep = []
        for x in samples:
            features = self.encoder(x)[-1]
            x = self.conv1(features)
            x = x.view(-1, 8, 10)
            last_rep.append(x)

        last_rep = torch.stack(last_rep)
        x = self.upsample_cls(last_rep)
        # x = nn.ReLU(self.convTr1(last_rep))
        # x = self.bn1(x)
        # x = nn.ReLU(self.convTr2(x))
        # x = self.bn2(x)
        # x = self.convTr3(x)
        x = torch.softmax(F.interpolate(x, (800,800)), dim=1)
        return x


class Bounding_box(nn.Module):

    def __init__(self):
        super().__init__()
        self.input_shape = (800,800)
        self.encoder = models.resnet50(pretrained=True)
        self.classifier = nn.Conv2d(75, 10, kernel_size=3, padding=1, bias=False)
        self.fc1 = nn.Linear(800, 1024)
        self.fc2 = nn.Linear(800, 1024*4)
        self.bn1 = nn.BatchNorm2d(10)

        self.regressor = nn.Conv2d(10, 4*4, kernel_size=3, padding=1, bias=False)
        self.pred = nn.Conv2d(10, 4*9, kernel_size=3, padding=1, bias=False)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=self.input_shape, mode='bilinear', align_corners=False)

        pred_x = self.pred(x)
        box_x = self.regressor(x)
        #print('Pred: ',pred_x.shape)
        #print('Box_x: ',box_x.shape)
        return pred_x, box_x
