import os
import torch
import random
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

class Segment_trainer():
    def __init__(self, model, optimizer, scheduler, epochs, trainloader, valloader, device="cpu", preT=False):
        self.model = model
        self.optim = optimizer
        self.epochs = epochs
        self.train_loader = trainloader
        self.val_loader = valloader
        self.sched = scheduler
        self.device = device
        self.scaleX = [100, 70, 50, 20]
        self.scaleY = [25, 20, 15, 5]
        self.map_sz = 800
        self.img_ht = 256
        self.img_wd = 306
        self.batch_size = 2

        if(preT):
            self.model.load_state_dict(torch.load('segment.pth', map_location=self.device))
        self.model = self.model.to(self.device)


    def step(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.sched.step()

 
    def loss_func(self, out, tar):
        loss = nn.functional.binary_cross_entropy(out, tar)
        return loss


    def tr(self, epoch):
        total_loss = 0
        for i, (sample, target, road_image, extra) in enumerate(self.train_loader):
            samples = torch.stack(sample).to(self.device)
            samples = samples.view(2, -1, 256, 306).to(self.device)
            road_image = torch.stack(road_image).type(dtype=torch.float32).to(self.device)

            out = torch.sigmoid(self.model(samples)).squeeze(1)
            #print(out.shape)
            loss = self.loss_func(out, road_image)
            self.step(loss)
            total_loss += loss.item()

            if i % 20  == 0:
                avg_loss = float(total_loss / (i+1))
                print('Epoch: {} | Avg Loss: {}'.format(epoch, avg_loss))

        avg_loss = float(total_loss / len(self.train_loader))
        print('Epoch {} | Train | Avg Loss: {}'.format(epoch, avg_loss))


    def validate(self, epoch):
        total_loss = 0
        with torch.no_grad():
            for i, (sample, target, road_image, extra) in enumerate(self.val_loader):
                samples = torch.stack(sample).to(self.device)
                samples = samples.view(2, -1, 256, 306).to(self.device)
                road_image = torch.stack(road_image).type(dtype=torch.float32).to(self.device)

                out = torch.sigmoid(self.model(samples)).squeeze(1)
                loss = self.loss_func(out, road_image)
                total_loss += loss.item()
                if i % 10  == 0:
                    avg_loss = float(total_loss / (i+1))
                    print('Validating Epoch: {} | Avg Loss: {}'.format(epoch, avg_loss))

            avg_loss = float(total_loss / len(self.val_loader))
            print('Validated Epoch {} | Total Avg Loss: {}'.format(epoch, avg_loss))
            torch.save(self.model.state_dict(), 'segment.pth')


    def train(self):
        print('Training...')
        for ep in range(self.epochs):
            self.tr(ep)
            self.val(ep)


    def plot(self, gt_truth, pred):
        plt.imshow(gt_truth)
        plt.show()
        plt.imshow(pred)
        plt.show()


    def visualize(self):
        with torch.no_grad():
            for i, (sample, target, road_image, extra) in enumerate(self.val_loader):
                samples = torch.stack(sample).to(self.device)
                samples = samples.view(2, -1, 256, 306).to(self.device)
                road_image = torch.stack(road_image).type(dtype=torch.float32)
                out = torch.sigmoid(self.model(samples)).squeeze(1)
                gt_truth = (road_image[0]).cpu().numpy()
                pred = out[0].cpu().numpy()
                self.plot(gt_truth, pred)
