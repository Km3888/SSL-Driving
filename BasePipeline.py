import torch.nn as nn
import torch
import torch.functional as F
import torchvision
import numpy as np

from provided_materials.code.data_helper import UnlabeledDataset,LabeledDataset
from provided_materials.code.helper import collate_fn

class BasePipeline(nn.module):

    def __init__(self,
                 encoder,
                 pretext_tasks:dict,
                 image_folder:str,
                 annotation_csv:str,
                 num_validation_scenes:int, #how many scenes we're holding out for validation
             ):
        super(BasePipeline,self).__init__()

        self.encoder=encoder
        self.pretext_tasks=pretext_tasks
        self.num_validation_scenes=num_validation_scenes
        self.image_folder=image_folder
        self.annotation_csv=annotation_csv

    def ssl_training(self):
        pass

    def supervised_training(self):
        pass

    def train(self):
        self.ssl_training()
        self.supervised_training()

    def get_data_loader(self,labeled,batch_size,first_dim=None,validation=False):
        transform = torchvision.transforms.ToTensor()

        if labeled:
            if validation:
                scene_index = np.arange(133 - self.num_validation_scenes, 134)
            else:
                scene_index = np.arange(106, 134 - self.num_validation_scenes)
            dataset = LabeledDataset(image_folder=self.image_folder,
                                              annotation_file=self.annotation_csv,
                                              scene_index=scene_index,
                                              transform=transform,
                                              extra_info=True
                                              )
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                                                      collate_fn=collate_fn)
        else:
            assert (first_dim is not None)
            scene_index = np.arange(106)
            dataset = UnlabeledDataset(image_folder=self.image_folder, scene_index=self.unlabeled_scene_index,
                                                  first_dim=first_dim, transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        return loader





if __name__=='__main__':
    base=BasePipeline(None,)