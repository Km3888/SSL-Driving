import torch
import torch.nn as nn


class DetectionModel(nn.Module):

    def __init__(self,encoder,grid_size=80,out_channels=5,encoder_weights=None):
        super(DetectionModel, self).__init__()
        self.encoder=encoder
        if encoder_weights:
            self.encoder.load_state_dict(torch.load(encoder_weights, map_location=self.device))

        self.fc1=nn.Linear(3072,)

        self.conv2d=nn.Conv2d(1,5,)


    def forward(self,batch):
        h=self.encoder(batch)

        h=self.fc1(h)
        h=h.view()
