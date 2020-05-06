import torch 
from box_calc import get_iou
import matplotlib.pyplot as plt 
import matplotlib.patches as pat
from collections import namedtuple
import csv
import io
import pdb

# works cited:
# inspiration and maybe some code from
# https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/1c5a58f47c9904a9fe9903b46c73bf93a230d8e7/models.py#L191-L200

# loss function against our yolo-style object detection labels
# if ground truth is that no objects are anchored in cell, 
# then loss is only calculated from that element.
# otherwise whole vector is calculated
mse_loss = torch.nn.MSELoss()
def yolo_loss_fn(true_label_tensor, predicted_tensor):
	assert(true_label_tensor.size() == predicted_tensor.size())

	# element wise differences
	squared_differences = (true_label_tensor - predicted_tensor).pow(2)

	# grid cells for which true label was 0
	cell_truly_empty_mask = true_label_tensor[:,:,0] == 0

	# for truly empty cells, aggreagte loss only from first element
	no_obj_loss = squared_differences[cell_truly_empty_mask][:,0].sum()

	# for other cells, aggregate total loss of whole cell
	yes_obj_loss = squared_differences[~cell_truly_empty_mask].sum()

	loss = no_obj_loss + yes_obj_loss

	return loss.item()

def test():
	lab = torch.ones([2,2,5])
	lab[0,0] = torch.zeros(5)
	pred = torch.ones([2,2,5])*3
	rez = yolo_loss_fn(lab, pred)
	# pdb.set_trace()

if __name__ == '__main__':
	test()
