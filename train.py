import os
import torch
import random
import matplotlib
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from seg_trainer import Segment_trainer
from model import RoadMap, BoundingBox
from helper import collate_fn, draw_box
from obj_detector import Object_det_trainer
from data_helper import UnlabeledDataset, LabeledDataset

# seeds
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
matplotlib.rcParams['figure.figsize'] = [5, 5]
matplotlib.rcParams['figure.dpi'] = 200

image_folder = '../data'
annotation_csv = '../data/annotation.csv'
labeled_scene_index = np.arange(106, 107)
labeled_scene_index_test = np.arange(130, 131)

transform = torchvision.transforms.ToTensor()
labeled_train_set = LabeledDataset(image_folder=image_folder,
                                  annotation_file=annotation_csv,
                                  scene_index=labeled_scene_index,
                                  transform=transform,
                                  extra_info=True
                                 )
labeled_val_set = LabeledDataset(image_folder=image_folder,
                                  annotation_file=annotation_csv,
                                  scene_index=labeled_scene_index_test,
                                  transform=transform,
                                  extra_info=True
                                 )

train_loader = torch.utils.data.DataLoader(labeled_train_set, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(labeled_val_set, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)
print('data loaded ')

def obj_det():
    num_epochs = 10
    obj_model = BoundingBox().double()
    optimizer_obj = torch.optim.SGD(
            obj_model.parameters(),
            lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler_obj = torch.optim.lr_scheduler.LambdaLR(
            optimizer_obj,
            lr_lambda=lambda x: (1 - x /( len(train_loader) * num_epochs)) ** 0.9)

    obj_trainer = Object_det_trainer(obj_model, optimizer_obj, scheduler_obj, num_epochs, train_loader, val_loader, device="cpu")
    obj_trainer.train()

def vis_obj_det():
    num_epochs = 10
    obj_model = BoundingBox().double()
    optimizer_obj = torch.optim.SGD(
            obj_model.parameters(),
            lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler_obj = torch.optim.lr_scheduler.LambdaLR(
            optimizer_obj,
            lr_lambda=lambda x: (1 - x /( len(train_loader) * num_epochs)) ** 0.9)

    obj_trainer = Object_det_trainer(obj_model, optimizer_obj, scheduler_obj, num_epochs, train_loader, val_loader, device="cpu", preT=True)
    obj_trainer.visualize()

def seg():
    num_epochs = 10
    seg_model = RoadMap()
    optimizer_seg = torch.optim.SGD(
            seg_model.parameters(),
            lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler_seg = torch.optim.lr_scheduler.LambdaLR(
            optimizer_seg,
            lr_lambda=lambda x: (1 - x /( len(train_loader) * num_epochs)) ** 0.9)

    seg_trainer = Segment_trainer(seg_model, optimizer_seg, scheduler_seg, num_epochs, train_loader, val_loader, device="cpu")
    seg_trainer.train()


def vis_seg():
    num_epochs = 10
    seg_model = RoadMap()
    optimizer_seg = torch.optim.SGD(
            seg_model.parameters(),
            lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler_seg = torch.optim.lr_scheduler.LambdaLR(
            optimizer_seg,
            lr_lambda=lambda x: (1 - x /( len(train_loader) * num_epochs)) ** 0.9)

    seg_trainer = Segment_trainer(seg_model, optimizer_seg, scheduler_seg, num_epochs, train_loader, val_loader, device="cpu", preT=True)
    seg_trainer.visualize()

#obj_det()
seg()
#vis_seg()
#vis_obj_det()