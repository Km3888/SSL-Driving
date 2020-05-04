from yolo_labeling import *
import test_data

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

def load_real_data():
	annot_scene0_lines = test_data.first_scene

	result = annotation_to_yolo_label(annot_scene0_lines, 3)
	print(result)

def main():
	test_corner_to_anchor()
	load_real_data()

if __name__ == '__main__':
	main()