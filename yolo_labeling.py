import torch 
from box_calc import get_iou
import matplotlib.pyplot as plt 
import matplotlib.patches as pat
from collections import namedtuple
import csv
import io

# raw data from annotations
YoloObj = namedtuple('YoloObj', ['x', 'y','height', 'width'])

# if box is not aligned with edges of image, 
# this creates larger bounding box that is
def corners_to_anchor(fl_x,fr_x,bl_x,br_x,fl_y,fr_y,bl_y,br_y):
    xs = [fl_x, fr_x, bl_x, br_x]
    ys = [fl_y, fr_y, bl_y, br_y]

    width = max(xs) - min(xs)
    height = max(ys) - min(ys)

    x = min(xs) + width/2.0
    y = min(ys) + height/2.0

    return YoloObj(x, y, height, width)

# todo not parsing string directly
def row_to_anchor(row_line):
    parsed_corners = [float(n) for n in row_line.split(",")[4:12]]
    return corners_to_anchor(*parsed_corners)


def anchor_to_cell_index(anchor, grid_dim, map_dim=80.0):
    map_offset = map_dim/2.0
    # translte coordinates to all positive
    x_grid_index = int((anchor.x + map_offset)/grid_dim)
    y_grid_index = int((anchor.y + map_offset)/grid_dim)
    return (x_grid_index, y_grid_index)

# assumes locations are wrt -40, 40 for both x and y
# normalized output between 0 and 1 with UPPER LEFT = 0,0 and BOTTOM_RIGHT = 1,1
# todo negative vs positive etc
def in_cell_loc(anchor, grid_dim, map_dim=80.0):
    map_offset = map_dim/2.0
    (cell_x, cell_y) = anchor_to_cell_index(anchor, grid_dim)

    residual_x = anchor.x+map_offset - cell_x*grid_dim
    residual_y = anchor.y+map_offset - cell_y*grid_dim

    return (residual_x/grid_dim, residual_y/grid_dim)

# for the objets present in a given sample, 
def annotation_to_yolo_label(sample_lines, n_grid_cells=19):
    # outputs labels in form consistent with YOLO algorithm
    # (s.t. there is only one anchor point per grid cell)
    # n_grid_cells is per dimension. used for both x and y.

    # output is tensor of dimension 
    # n_grid_cells x n_grid_cells x 5
    # with 5-length vector is of format:
    # is_object_present, x, y, height, width
    # (where x and y are between 0 and 1)

    # assumes sample input locations are bounded by -40, 40

    grid_dim = 80.0/n_grid_cells

    output = torch.zeros([n_grid_cells, n_grid_cells, 5])
    yolo_objs = [row_to_anchor(row) for row in sample_lines]

    cell_indices = [anchor_to_cell_index(anchor, grid_dim) for anchor in yolo_objs]
    cell_normalized_locations = [in_cell_loc(anchor, grid_dim) for anchor in yolo_objs]

    # todo if there are multiple anchors in one grid, print warning
    # signals we need higher resolution grid

    for (anchor, cell_index, in_cell_xy) in zip(yolo_objs, cell_indices, cell_normalized_locations):
        (cell_x, cell_y) = cell_index
        (local_x, local_y) = in_cell_xy

        positive_vector = torch.tensor([1, local_x, local_y, anchor.height/grid_dim, anchor.width/grid_dim])
#        print("DB positive_vector", positive_vector)
#        print("DB cellx,celly: ", cell_x, cell_y)
        output[cell_x, cell_y, :] = positive_vector

    return output








