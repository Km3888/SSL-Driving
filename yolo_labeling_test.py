from yolo_labeling import *
import test_data
import numpy as np
from provided_materials.code.data_helper import UnlabeledDataset,LabeledDataset
from provided_materials.code.helper import collate_fn
import torch
import torchvision


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

# def load_real_data():
# 	annot_scene0_lines = test_data.first_scene

# 	result = annotation_to_yolo_label(annot_scene0_lines, 3)
# 	print(result)

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
	import pdb


	for target in targets:
		cur_res = dl_target_tuple_as_yolo_tensor(target)
		# print(cur_res)
		if len(target) > 0:
			# there should at least be some non zero elements of grid
			assert(sum(sum(sum(cur_res))) > 0)
	# pdb.set_trace()
	print('pased run_w_dataloader')


def main():
	test_corner_to_anchor()
#	load_real_data()
	run_w_dataloader()

if __name__ == '__main__':
	main()