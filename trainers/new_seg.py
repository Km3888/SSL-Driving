import os
import math
import torch
import random
import matplotlib
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F


class NewSegTrainer():
    def __init__(self, model, optim, sched, epochs, train_loader, val_loader, batch_size, device="cpu", preT=False):
        self.model = model
        self.optim = optim
        self.epochs = epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.sched = sched
        self.device = device
        self.batch_size = batch_size
        if(preT):
            #load with optim para if continuing training
            self.model.load_state_dict(torch.load('new_seg_model.pth', map_location=self.device))
        self.model = self.model.to(self.device)
        self.map_sz = 800
        self.img_h = 256
        self.img_w = 306


    def plotMap(x, y):
        plt.ylim(800, 0) #decreasing order
        plt.plot(x.numpy(), y.numpy(), 'o', color='black')
        plt.show()

    def take_step(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()


    def tr(self, epoch):
        total_loss = 0
        b_loss = 0
        print('Training ...')
        for i, (sample, target, road_image, extra) in enumerate(self.train_loader):
            samples = torch.stack(sample).to(self.device).float()
            road_image = torch.stack(road_image).type(dtype=torch.long).to(self.device)
            out = self.model(samples).squeeze(1)
            loss = nn.functional.cross_entropy(out, road_image)
            self.take_step(loss)
            total_loss += loss.item()
            
            if i % 20  == 0 or True:
                avg_loss = float(total_loss / (i+1))
                print('Training epoch: {} | Mean classification loss: {}'.format(epoch, avg_loss))
            torch.cuda.empty_cache()
        
    def val(self, epoch):
        with torch.no_grad():
            total_loss = 0
            b_loss = 0
            print('Validation')
            for i, (sample, target, road_image, extra) in enumerate(self.val_loader):
                samples = torch.stack(sample).to(self.device).float()
                road_image = torch.stack(road_image).type(dtype=torch.long).to(self.device)
                out = self.model(samples).squeeze(1)
                loss = nn.functional.cross_entropy(out, road_image)
                total_loss += loss.item()
                
            avg_loss = float(total_loss / len(self.val_loader))
            self.sched.take_step(avg_loss)
            print('Val ppoch: {} | Mean classification Loss: {}'.format(epoch, avg_loss))
            torch.cuda.empty_cache()
            state = {'epoch': epoch + 1, 'state_dict': self.model.state_dict(),'optim': self.optim.state_dict()}
            torch.save(state, 'new_segment_full.pth')
            torch.save(self.model.state_dict(), 'new_seg_model.pth')

    def getHeatMap(self, classes):
        gt_widths = self.gt_boxes[:, 2] - self.gt_boxes[:, 0]
        gt_heights = self.gt_boxes[:, 3] - self.gt_boxes[:, 1]
        gt_center_x = self.gt_boxes[:, 0] + 0.5*gt_widths
        gt_center_y = self.gt_boxes[:, 1] + 0.5*gt_heights
        
        gt_heat_map_x = gt_center_x[classes > 0]
        gt_heat_map_y = gt_center_y[classes > 0]

        plotMap(gt_heat_map_x, gt_heat_map_y)
    
    def plot(self, gt_truth, pred):
        matplotlib.rcParams['figure.figsize'] = [5, 5]
        matplotlib.rcParams['figure.dpi'] = 200
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(gt_truth)
        axs[1].imshow(pred)
        plt.show()


    def train(self):
        print('Training...')
        for ep in range(self.epochs):
            self.tr(ep)
            self.val(ep)


    def visualize(self):
        with torch.no_grad():
            for i, (sample, target, road_image, extra) in enumerate(self.val_loader):
                samples = torch.stack(sample).to(self.device).float()
                road_image = torch.stack(road_image).type(dtype=torch.long).to(self.device)
                gt_truth = (road_image[0]).cpu().numpy()

                out = self.model(samples).squeeze(1)
                pred = torch.argmax(out[0], dim=0).cpu().numpy()
                self.plot(gt_truth, pred)