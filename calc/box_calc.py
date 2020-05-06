import torch
from torch.jit.annotations import Tuple
from torch import Tensor
import torchvision

def rect_area(rectangles):
    '''
    Args: rectangles (Tensor[n, 4])(x1, y1, x2, y2): coordinates of the box whose area is returned
    Returns: (Tensor[n]): area of boxes
    '''
    return (rectangles[:, 2] - rectangles[:, 0]) * (rectangles[:, 3] - rectangles[:, 1])


# modified from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
def get_iou(b1, b2):
    '''
    Args:
        b1: (Tensor[n, 4])
        b2: (Tensor[m, 4])
    Returns:
        iou (Tensor[n, m]): nxm pairwise IoU values for each b1 and b2
    '''
    a1 = rect_area(b1)
    a2 = rect_area(b2)

    l = torch.max(b1[:, None, :2], b2[:, :2])  # of size [n,m,2]
    r = torch.min(b1[:, None, 2:], b2[:, 2:]) 

    w = (r - l).clamp(min=0)  # of size [n,m,2]
    intersect = w[:, :, 0] * w[:, :, 1]  # # of size [n,m]

    iou = intersect / (a1[:, None] + a2 - intersect).type(torch.double)
    return iou

'''
def test_iou():
    b1 = torch.tensor([2,2,5,5]).view(1,-1)
    b2 = torch.tensor([1,1,3,3]).view(1,-1)
    print(get_iou(b1, b2)) 

test_iou()
'''