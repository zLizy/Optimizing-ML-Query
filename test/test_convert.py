# file name: test.py
import unittest
import evaluate from evaluation

class TestModule(unittest.TestCase):

    @classmethod
	def setUpCalss(cls):
		print('setUpClass')
        self.test = True
        self.all = False
        self.run = 10
        self.data = 'coco' # coco voc
        if self.data == 'coco':
            self.gt_file = '/home/zli/experiments/datasets/coco/data/annotations/val2017.csv'
        elif self.data == 'voc':
            self.gt_file = '/home/zli/experiments/datasets/voc_2012/gt.csv'

        level='high'
            if self.data == 'voc':
                self.config_file = '../convert/voc_model_config_new_model_30_'+level+'.csv'
            else:
                self.config_file = '../config/coco_model_config_new_model_30_'+level+'.csv'
	
	# only execute once in the end
	@classmethod
	def tearDownClass(cls):
		print('tearDownClass')


	# All the function name shall name beginned with 'test'
	def test_class_name(self):
		categories_coco = [
        'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
        'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
        'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
        ]
		self.assertEqual(result,X)

	
    def test_file_name(self):
		if  test:
            base_file = 'base_'+data+'_test.csv'
            val_file = 'raw_'+data+'_test'
            out_dir = 'inference_'+data+'_'+level
        elif all:
            # evaluate all
            base_file = 'base_all.csv'
            val_file = 'raw_all'
            metric_dir = 'metrics_all_'+level
            selectivity_dir = 'selectivity_all_'+level
            out_dir = ['val_all_'+level,metric_dir,selectivity_dir]
        elif not test:
            base_file = 'base_'+data+'_val.csv'
            val_file = 'raw_'+data+'_val'
            metric_dir = 'metrics_'+data+'_'+level
            selectivity_dir = 'selectivity_'+data+'_'+level
            out_dir = ['val_'+data+'_'+level,metric_dir,selectivity_dir]


    

if __name__ == '__main__':
	unittest.main()