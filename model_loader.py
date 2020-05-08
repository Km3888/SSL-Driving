"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models.new_model import NewRoad_map #Road_map

# import your model class
# import ...

# Put your transform function here, we will use it for our dataloader
def get_transform(): 
    # return torchvision.transforms.Compose([
    # 
    # 
    # ])
    pass

class ModelLoader():
    # Fill the information for your team
    team_name = 'something'
    team_member = ["Kelly Marshall", "Daniel Jefferys-White", "Poornima Haridas"]
    contact_email = '@nyu.edu'

    def __init__(model_file):
        # You should 
        #       1. create the model object
        #       2. load your state_dict
        #       3. call cuda()
        # self.model = ...
        # 
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        #self.bbox_model = 
        #self.bbox_model.load_state_dict(torch.load('model1.pth', map_location=self.device))
        self.road_model = NewRoad_map() # or Road_map()
        self.road_model.load_state_dict(torch.load('new_seg_model.pth', map_location=self.device))
        #self.gt_boxes = 

    def get_bounding_boxes(samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object

        pass

    def get_binary_road_map(samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800] 
        
        out = self.road_model(samples).squeeze(1)
        road_mask = torch.argmax(out, dim=1)
        return road_mask
