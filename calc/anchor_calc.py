import torch 
from calc.box_calc import get_iou
import matplotlib.pyplot as plt


def calc_offset(pred_boxes, og_boxes):
    og_w = og_boxes[:, 2] - og_boxes[:, 0]
    og_h = og_boxes[:, 3] - og_boxes[:, 1]
    og_cent_x = og_boxes[:, 0] + 0.5*og_w
    og_cent_y = og_boxes[:, 1] + 0.5*og_h

    pb_w = pred_boxes[:, 2] - pred_boxes[:, 0]
    pb_h = pred_boxes[:, 3] - pred_boxes[:, 1]
    pb_cent_x = pred_boxes[:, 0] + 0.5*pb_w
    pb_cent_y = pred_boxes[:, 1] + 0.5*pb_h

    del_x = (pb_cent_x - og_cent_x) / og_w
    del_y = (pb_cent_y - og_cent_y) / og_h
    del_x_scaled = torch.log(pb_w / og_w)
    del_y_scaled = torch.log(pb_h / og_h)

    offsets = torch.cat([del_x.unsqueeze(0),del_y.unsqueeze(0),
                    del_x_scaled.unsqueeze(0),del_y_scaled.unsqueeze(0)],dim=0)
    return offsets.permute(1,0)

def get_og_bboxes(bboxes, classes, pred_boxes, size, device):
    max_thresh = 0.75
    min_thresh = 0.33
    bboxes = bboxes.clone()
    bboxes *= 10
    bboxes = bboxes + 400
    classes += 1
    ex1 = bboxes[:, 0, 3].unsqueeze(0)
    ey1 = bboxes[:, 1, 3].unsqueeze(0)
    ex2 = bboxes[:, 0, 0].unsqueeze(0)
    ey2 = bboxes[:, 1, 0].unsqueeze(0)
    og_boxes = torch.cat([ex1, ey1, ex2, ey2], dim=0)
    og_boxes = og_boxes.permute(1,0)

    IoUs = get_iou(pred_boxes, og_boxes)
    v, ind = torch.max(IoUs, dim=1)

    og_w = og_boxes[:, 2] - og_boxes[:, 0]
    og_h = og_boxes[:, 3] - og_boxes[:, 1]
    og_cent_x = og_boxes[:, 0] + 0.5*og_w
    og_cent_y = og_boxes[:, 1] + 0.5*og_h
    
    pb_ws = pred_boxes[:, 2] - pred_boxes[:, 0]
    pb_hs = pred_boxes[:, 3] - pred_boxes[:, 1]
    pb_cent_x = pred_boxes[:, 0] + 0.5*pb_ws
    pb_cent_y = pred_boxes[:, 1] + 0.5*pb_hs

    og_classes = torch.zeros((size*size*4)).type(torch.long).to(device)
    og_classes[v < min_thresh] = 0 # bg anchors
    og_classes[v > max_thresh] = classes[ind[v > max_thresh]] # fg anchors
    og_classes[(v >= min_thresh) & (v < max_thresh)] = -1 # ignore

    og_offsets = torch.zeros((size*size*4, 4)).type(torch.double).to(device) 
    original_boxes = og_boxes[ind[v > max_thresh]]
    predicted_boxes = pred_boxes[v > max_thresh]
    offsets = calc_offset(predicted_boxes, original_boxes)
    og_offsets[v > max_thresh] = offsets

    return og_classes, og_offsets

def plot_base(x, y):
    plt.ylim(800, 0) # time
    plt.plot(x.numpy(), y.numpy(), 'o', color='black')
    plt.show()