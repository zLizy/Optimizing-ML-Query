# file name: test.py
import unittest
import sys
import pandas as pd
import glob
sys.path.append('../')
from inference import evaluate 

class TestEvaluation(unittest.TestCase):

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
    
    def setUp(self):
        print('setUp')
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

    def tearDown(self):
        print('tearDown\n')		

    # verify whether the correct image files are retrieved
    def test_img_file(self):
        # test
        test = True
        val_file = '../evaluation/raw_'+self.data+'_test'
        img_files = pd.read_csv(glob.glob(val_file+'/summary_1_yolov3_*.csv')[0],index_col=0)['filename']
        test_img_files = pd.read_csv('../evaluation/base_coco_test.csv')['filename']
        self.assertCountEqual(img_files, test_img_files)
        
        # validation
        test = False
        base_file = 'base_'+self.data+'_val.csv'
        val_file = '../evaluation/raw_'+self.data+'_val'
        # print(val_file+'/summary_1_yolov3_*.csv')
        img_files = pd.read_csv(glob.glob(val_file+'/summary_1_yolov3_*.csv')[0],index_col=0)['filename']
        test_img_files = pd.read_csv('../evaluation/base_coco_val.csv')['filename']
        self.assertCountEqual(img_files, test_img_files)

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
        nms = ['person','bicycle','car','motorcycle','airplane','bus','train',\
                    'truck','boat','traffic_light','fire_hydrant','stop_sign','parking_meter',\
                    'bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra',\
                    'giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis',\
                    'snowboard','sports_ball','kite','baseball_bat','baseball_glove','skateboard',\
                    'surfboard','tennis_racket','bottle','wine_glass','cup','fork','knife','spoon',\
                    'bowl','banana','apple','sandwich','orange','broccoli','carrot','hot_dog','pizza',\
                    'donut','cake','chair','couch','potted_plant','bed','dining_table','toilet','tv',\
                    'laptop','mouse','remote','keyboard','cell_phone','microwave','oven','toaster',\
                    'sink','refrigerator','book','clock','vase','scissors','teddy_bear','hair_drier','toothbrush'
            ]

        self.assertListEqual([x.replace(' ','_') for x in categories_coco if x !='N/A'], nms)

	
    def test_file_name(self):
        pass

        # self.assertEqual(result,X)
    

if __name__ == '__main__':
    unittest.main()