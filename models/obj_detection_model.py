import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionModel(nn.Module):

    def __init__(self,encoder,grid_size=80,out_channels=5,encoder_weights=None):
        super(DetectionModel, self).__init__()
        self.encoder=encoder
        if encoder_weights:
            self.encoder.load_state_dict(torch.load(encoder_weights, map_location=self.device))

        self.fc1=nn.Linear(128*6,1805)

        #TODO: Add convolutional/deconvolutional layers


    def forward(self,batch):
        batch_size,_,_,_,_=batch.size()
        batch=batch.view(batch_size*6,3,256,306).float()
        h,z=self.encoder(batch)
        z=z.view(batch_size,128*6)
        output=self.fc1(z)
        output=output.view(batch_size,19,19,5)
        output[:,:,:,0]=torch.sigmoid(output[:,:,:,0])
        output[:,:,:,1:]=(80/19)*torch.sigmoid(output[:,:,:,1:])
        return output
