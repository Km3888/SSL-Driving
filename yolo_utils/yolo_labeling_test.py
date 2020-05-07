from yolo_utils.yolo_labeling import *
import numpy as np
from provided_materials.code.data_helper import LabeledDataset
from provided_materials.code.helper import collate_fn
import torch
import torchvision
import pdb
from numpy.testing import assert_approx_equal


def test_corner_to_anchor():
    # simple box
    result = corners_to_anchor(5,11,5,11, 5, 5, 1, 1)
    assert(result.x == 8 
        and result.y == 3 
        and result.height == 4 
        and result.width == 6)
    # angled box
    result = corners_to_anchor(5,11,12,18,10,14,2,6)
    assert(result.x == (18+5)/2.0 
        and result.y == (14+2)/2.0
        and result.height == (14-2)
        and result.width == (18-5))
    print('passed test_corner_to_anchor')

def run_w_dataloader():
	labeled_scene_index = np.arange(106, 134)
	dataset = LabeledDataset(image_folder='provided_materials/data',
	                         annotation_file='provided_materials/data/annotation.csv',
	                         scene_index=labeled_scene_index,
	                         transform=torchvision.transforms.ToTensor(),
	                         extra_info=True)
	loader = torch.utils.data.DataLoader(dataset,
	                                     batch_size=4,
	                                     shuffle=True,
	                                     num_workers=2,
	                                     collate_fn=collate_fn)
	sample, targets, road_image, extra=iter(loader).next()



	for target in targets:
		cur_res = dl_target_tuple_as_yolo_tensor(target['bounding_box'])
		# print(cur_res)
		if len(target) > 0:
			# there should at least be some non zero elements of grid
			assert(sum(sum(sum(cur_res))) > 0)
	# pdb.set_trace()
	print('pased run_w_dataloader')

def test_dl_target_tuple_as_yolo_tensor():
	# just one element at a known location- box in upper right quadrant
	# fl_x,fr_x,bl_x,br_x,fl_y,fr_y,bl_y,br_y

	test_samp_target = torch.tensor(
		[[5, 10, 5, 10], 
		[25,25,15,15]], 
		dtype=torch.float64
		)
	
	res3 = dl_target_tuple_as_yolo_tensor(tuple([test_samp_target]), 3)
#	pdb.set_trace()	
	print('res3[1][2][1]', res3[1][2][1])
	DIM = (80/3)
	assert_approx_equal(res3[1][2][1] * DIM + 2*DIM, (40+((10+5)/2)))
#	assert_approx_equal(res3[0][0][2], ((40+((25+15)/2)) / 80 - DIM)/DIM)
	
	res1 = dl_target_tuple_as_yolo_tensor(tuple([test_samp_target]), 1)
	print('res1', res1)
	# centroid at x = (40+((10+5)/2) / 80)
	# centroid at y = (40+((25+15)/2) / 80)
	assert_approx_equal(res1[0][0][1], (40+((10+5)/2)) / 80)
	assert_approx_equal(res1[0][0][2], (40+((25+15)/2)) / 80)	

	assert_approx_equal(2,6)


def test_yolo_to_bbox():

	labeled_scene_index = np.arange(106, 134)
	dataset = LabeledDataset(image_folder='provided_materials/data',
	                         annotation_file='provided_materials/data/annotation.csv',
	                         scene_index=labeled_scene_index,
	                         transform=torchvision.transforms.ToTensor(),
	                         extra_info=True)
	loader = torch.utils.data.DataLoader(dataset,
	                                     batch_size=4,
	                                     shuffle=True,
	                                     num_workers=2,
	                                     collate_fn=collate_fn)
	sample, targets, road_image, extra=iter(loader).next()	
	for target in targets:
		bb = target['bounding_box']
#		pdb.set_trace()
		cur_res = dl_target_tuple_as_yolo_tensor(target['bounding_box'])
		# print(cur_res)
#		[[-5.,10.,-5.,10.],[3.,3.,-6.,-6.]],		

	# fl_x,fr_x,bl_x,br_x,
	# fl_y,fr_y,bl_y,br_y
	tens_w_right_facing_mock = torch.tensor([
		[[ 10,10,-5,-5],[3,-6,3,-6]]
        ], 
        dtype=torch.float64)


	# todo could make prop test w config above

	NCELL = 10

	px = tens_w_right_facing_mock

	pxyo = dl_target_tuple_as_yolo_tensor(tens_w_right_facing_mock, NCELL)

	pxre = yolo_label_to_bbox_format(pxyo, NCELL)
	print('px',px)
	print('pxre', pxre)
	pdb.set_trace()	

	# should be able to perfectly reconstruct non-angled bboxes
	# though potentially oriented differently

	# todo have to account for orientation change
#	print("HHHHHHHHHHHH")
	pdb.set_trace()	
	are_eq = torch.eq(tens_w_right_facing_mock, 
		yolo_label_to_bbox_format(
			dl_target_tuple_as_yolo_tensor(
				tens_w_right_facing_mock))
		)

	assert(torch.all(are_eq))

	# angled ones todo
	tens_w_angled_mock = torch.tensor([
		[[5,11,12,18],[10,14,2,6]]
        ], 
        dtype=torch.float64)


def main():
	test_corner_to_anchor()
#  todo:
#	test_dl_target_tuple_as_yolo_tensor()	
	run_w_dataloader()
	test_yolo_to_bbox()


if __name__ == '__main__':
	main()