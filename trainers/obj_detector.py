import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from calc.anchor_calc import get_og_bboxes, plot_base
from models.resnet_simclr import ResNetSimCLR

from provided_materials.code.data_helper import LabeledDataset
from provided_materials.code.helper import collate_fn

class Object_det_trainer():
    def __init__(self, model, optim, scheduler, epochs, trainloader, valloader, device="cpu", preT=False):
        self.model = model
        self.optim = optim
        self.epochs = epochs
        self.train_loader = trainloader
        self.val_loader = valloader
        self.sched = scheduler
        self.device = device
        self.map_size = 800
        self.x_scale = [100, 70, 50, 20]
        self.y_scale = [25, 20, 15, 5]
        self.img_ht = 256
        self.img_wd = 306
        self.batch_size = 2
        self.og_boxes = self.get_og_boxes()

        if(preT):
            self.model.load_state_dict(torch.load('classif.pth', map_location=self.device))
        self.model = self.model.to(self.device)


    def get_og_boxes(self):
        wds = torch.tensor(self.x_scale)
        hts = torch.tensor(self.y_scale)
        ref_boxes = []
        for x in range(self.map_size):
            for y in range(self.map_size):
                x_l = torch.tensor([x, x, x, x])
                y_l = torch.tensor([y, y, y, y])
                x_r = wds + x
                y_r = hts + y
                x_r = x_r.unsqueeze(0)
                y_r = y_r.unsqueeze(0)
                x_l = x_l.unsqueeze(0)
                y_l = y_l.unsqueeze(0)
                ref_box = torch.cat((x_l, y_l, x_r, y_r))
                ref_box = ref_box.permute((1,0))
                ref_boxes.append(ref_box)
    
        og_boxes = torch.stack(ref_boxes).view(-1,4).type(torch.double).to(self.device)
        return og_boxes


    def get_targets(self, target):
        batched_preds = []
        batched_offsets = []
        for t in target:
            bboxes = t['bounding_box'].to(self.device)
            og_classes, og_offsets = get_og_bboxes(bboxes, t['category'].to(self.device), self.og_boxes, self.map_size, self.device)
            batched_preds.append(og_classes)
            batched_offsets.append(og_offsets)
        target_class = torch.stack(batched_preds)
        target_box = torch.stack(batched_offsets)
        return target_class, target_box


    def step(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()


    def cls_loss(self, out_pred, target_class, val=False):
        # get equal number of +ve and -ve anchors
        out_pred = out_pred.permute((0,2,1))

        if val == True:
            good_targets = target_class[target_class != -1]
            good_preds = out_pred[target_class != -1]
            if good_preds.shape[0] == 0:
                return torch.tensor(0)
            return F.cross_entropy(good_preds, good_targets)

        bad_examples = target_class[target_class == -1]
        foreground_examples = target_class[target_class > 0]
        foreground_preds = out_pred[target_class > 0]
        background_examples = target_class[target_class == 0]
        background_preds = out_pred[target_class == 0]

        num_pos = foreground_examples.shape[0]

        if num_pos == 0:
            print('No positive anchors!!')
            return torch.tensor(0)

        perm1 = torch.randperm(background_examples.shape[0], device=self.device)[:]
        background_examples = background_examples[perm1]
        background_preds = background_preds[perm1]

        targets = torch.cat((background_examples, foreground_examples), dim=0)
        preds = torch.cat((background_preds, foreground_preds), dim=0)
        return F.cross_entropy(preds, targets)


    def bbox_loss(self, target_box, target_class, out_bbox):
        inds = (target_class != 0)
        target_box = target_box[inds]
        out_bbox = out_bbox[inds]
        loss_bbox = F.smooth_l1_loss(out_bbox, target_box)
        return loss_bbox


    def tr(self, epoch):
        total_loss = 0
        b_loss = 0
        for i, (sample, target, road_image, extra) in enumerate(self.train_loader):
            samples = torch.stack(sample).to(self.device).double()
            samples = samples.view(self.batch_size, 6, -1,self.img_ht, self.img_wd).to(self.device).double()

            target_class, target_box = self.get_targets(target)
            print(target_class.size())
            print(target_box.size())
            out_pred, out_bbox = self.model(samples)
            #print(target_class.size(), out_pred.size())
            out_bbox = out_bbox.view(self.batch_size, -1, 4)
            out_pred = out_pred.view(self.batch_size, 9, -1)
            
            loss_cls = self.cls_loss(out_pred, target_class)
            loss_bbox = self.bbox_loss(target_box, target_class, out_bbox)
            loss = loss_cls + loss_bbox
            if loss.item() != 0:
                self.step(loss)

            total_loss += loss_cls.item()
            b_loss += loss_bbox.item()

            if i % 20  == 0:
                avg_loss = float(total_loss / (i+1))
                avg_b_loss = float(b_loss / (i+1))
                print('Epoch {} | Avg classification loss= {}, BBox loss= {}'.format(epoch, avg_loss, avg_b_loss))
            torch.cuda.empty_cache()
        
        print('Epoch {} | Train | Cls loss: {} | BBox Loss: {}'.format(epoch, avg_loss, avg_b_loss))

    def val(self, epoch):
        # only classification rn
        total_loss = 0
        with torch.no_grad():
            for i, (sample, target, road_image, extra) in enumerate(self.train_loader):
                samples = torch.stack(sample).to(self.device)
                samples = samples.view(2, -1, self.img_ht, self.img_wd).to(self.device).double()

                target_class, target_box = self.get_targets(target)
                
                out_pred, out_bbox = self.model(samples)
                out_bbox = out_bbox.view(self.batch_size, -1, 4)
                out_pred = out_pred.view(self.batch_size, 9, -1)

                loss = self.cls_loss(out_pred, target_class, val=True)
                total_loss += loss.item()
                torch.cuda.empty_cache()
            avg_loss = float(total_loss)
            print('Epoch {} | Val | Total Avg Loss: {}'.format(epoch, avg_loss))
        torch.save(self.model.state_dict(), 'classif.pth')

    def train(self):
        for ep in range(self.epochs):
            self.tr(ep)
            self.val(ep)

    def getHeatMap(self, classes):
        og_wds = self.og_boxes[:, 2] - self.og_boxes[:, 0]
        og_hts = self.og_boxes[:, 3] - self.og_boxes[:, 1]
        og_center_x = self.og_boxes[:, 0] + 0.5*og_wds
        og_center_y = self.og_boxes[:, 1] + 0.5*og_hts
        
        og_heat_map_x = og_center_x[classes > 0]
        og_heat_map_y = og_center_y[classes > 0]
        #print(og_heat_map_x.shape, og_heat_map_y.shape)
        plot_base(og_heat_map_x, og_heat_map_y)

    def visualize(self):
        for i, (sample, target, road_image, extra) in enumerate(self.train_loader):
            samples = torch.stack(sample).to(self.device)
            samples = samples.view(2, -1, self.img_ht, self.img_wd).to(self.device).double()

            target_class, target_box = self.get_targets(target)
            out_pred, out_bbox = self.model(samples)
            out_bbox = out_bbox.view(self.batch_size, -1, 4)
            out_pred = out_pred.view(self.batch_size, 9, -1)
            out_pred = out_pred.permute((0,2,1))[0]

            out_scores, out_inds = torch.max(out_pred, dim=1)
            '''
            print(out_scores.size())
            print((out_scores < 0.9).size())
            print(out_inds.size())
            '''
            out_inds[out_scores < 0.9] = 0

            self.getHeatMap(target_class[0])
            self.getHeatMap(out_inds)


if __name__=='__main__':
    encoder=ResNetSimCLR('resnet18',1)
    obj_model=object_detection_model(encoder)
    num_epochs = 10
    train_index = np.arange(106, 125)
    train_dataset = LabeledDataset(image_folder='provided_materials/data',
                                   annotation_file='../provided_materials/data/annotation.csv',
                                   scene_index=train_index,
                                   transform=torchvision.transforms.ToTensor(),
                                   extra_info=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                         batch_size=2,
                                         shuffle=True,
                                         num_workers=2,
                                         collate_fn=collate_fn)
    val_index = np.arange(125,134)
    val_dataset = LabeledDataset(image_folder='provided_materials/data',
                                 annotation_file='../provided_materials/data/annotation.csv',
                                 scene_index=val_index,
                                 transform=torchvision.transforms.ToTensor(),
                                 extra_info=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=2,
                                         shuffle=True,
                                         num_workers=2,
                                         collate_fn=collate_fn)
    optimizer_obj = torch.optim.SGD(
        obj_model.parameters(),
        lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler_obj = torch.optim.lr_scheduler.LambdaLR(
        optimizer_obj,
        lr_lambda=lambda x: (1 - x / (len(train_loader) * num_epochs)) ** 0.9)

    obj_trainer = Object_det_trainer(obj_model, optimizer_obj, scheduler_obj, num_epochs, train_loader, val_loader,
                                     device="cpu")
    obj_trainer.tr(0)