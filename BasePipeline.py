import torch.nn as nn
import torch
import torchvision
import numpy as np

from provided_materials.code.data_helper import UnlabeledDataset,LabeledDataset

from models import oft_resnet


class BasePipeline(nn.Module):

    def __init__(self,
                 encoder,
                 image_folder:str,
                 annotation_csv:str,
                 num_validation_scenes:int, #how many scenes we're holding out for validation
             ):
        super(BasePipeline,self).__init__()

        self.encoder=encoder
        self.num_validation_scenes=num_validation_scenes
        self.image_folder=image_folder
        self.annotation_csv=annotation_csv

    def ssl_training(self):
        pass

    def supervised_training(self):
        pass

    def encode(self):
        image_loader=self.get_data_loader(labeled=False,batch_size=16,first_dim='image')
        sample=iter(image_loader).next()[0]
        code=self.encoder(sample)

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
        else:
            assert (first_dim is not None)
            scene_index = np.arange(106)
            dataset = UnlabeledDataset(image_folder=self.image_folder, scene_index=scene_index,
                                                  first_dim=first_dim, transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        return loader





if __name__=='__main__':
    encoder= oft_resnet.resnet18_encoder(pretrained=False, output_channels=32)
    base=BasePipeline(encoder,{},'provided_materials/data','provided_materials/data/annotation.csv',1)
    base.encode()