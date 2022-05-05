import numpy as np
import pandas as pd
import yaml
import os
import math

def write2file(model_info,latency,prob,path='model_config.csv'):
	if os.path.isfile(path):
		df = pd.read_csv(path,index_col=0)
		frames = [df]
	else:
		frames = []

	num_model = len(model_info)
	count = len(latency) // num_model
	df_new = pd.DataFrame(columns=['index','model','root','type','latency','probability'])
	
	type_list = []
	root_list = []
	model_list = []
	for key, value in model_info.items():
		model_list.append([key]*count)
		root_list.append([value[0]]*count)
		type_list.append([value[1]]*count)
	# print(root_list)
	# print(type_list)
	# model_list = [[m] * count for m in model_list]
	df_new['model'] = list(np.concatenate(model_list).flat) 
	df_new['root'] = list(np.concatenate(root_list).flat)
	df_new['type'] = list(np.concatenate(type_list).flat)
	df_new['latency'] = latency
	df_new['probability'] = [round(p,3) for p in prob]
	frames.append(df_new)

	results = pd.concat(frames,ignore_index=True)
	results['index'] = range(1,len(results)+1)
	
	print(results.head())

	results.to_csv(path)


def changeValues(col,val,path):
	if os.path.isfile(path):
		df = pd.read_csv(path,index_col=0)
	df[col] = val
	df.to_csv(path)

def convert(num_model,data):
	
	# model_name = data['model']
	count = int(data['count'])*num_model
	seed = int(data['seed'])
	l_min = data['latency_min'] # math.log10(data['latency_min'])
	l_max = data['latency_max'] # math.log10(data['latency_max'])
	p_min = data['prediction_min']
	p_max = data['prediction_max']
	print(count, l_min,l_max,p_min,p_max)

	np.random.seed(seed)

	latency = np.random.randint(low=l_min,high=l_max,size=count)
	
	# if model_name == 'yolov3':
	# 	rand_idx = np.random.randint(count, size=count//2)
	# 	latency = [l+1 if i in rand_idx else l for i,l in enumerate(latency)]

	print(latency)

	prob = np.random.uniform(low=p_min,high=p_max,size=count)
	# print(prob)

	return latency,prob

def set_task_coverage(data,count,model_info,coverage,categories):
	# fix number of task coverage
	choice = []
	coverage_list = []
	for model_name, attributes in model_info.items():
		for i in range(count):
			if attributes[1] == 'Seg':
				choice.append(categories[1])
				coverage_list.append(20)
			else:
				choice.append(list(np.random.choice(categories[0],size=coverage)))
				coverage_list.append(coverage)
		# choice = [np.random.choice(max_num+1,size=coverage) for i in range(len(df))]
	# print([max(c) for c in choice])

	return coverage_list, choice

def set_model_coverage(data,count,model_info,coverage,categories):
	
	if data == 'voc':
		categories = categories[1]
	else:
		categories = categories[0]
	size = len(model_info)*count
	df = pd.DataFrame(columns=categories,index=range(size))
	
	
	for i,category in enumerate(categories):
		# print(list(np.random.choice(range(size),size=coverage)))
		df.loc[list(np.random.choice(range(size),size=coverage)),category] = 1

	choices = []
	coverage = []
	for i in range(size):
		tasks = list(df.iloc[i,:].dropna().index)
		choices.append(tasks)
		coverage.append(len(tasks))
	

	return coverage,choices


def set_coverage(data,path,model_info,config,coverage=20,new_path='',categories=[]):
	# old_path,model_info,config,coverage=coverage,new_path=new_path,categories=[categories_coco,categories_seg]

	# path = 'model_config.csv'
	# path = '../repository/model_config_task_half_all.csv'
	df = pd.read_csv(path,index_col=0)

	count = int(config['count'])
	# max_num = 79
	# power law: most of the tasks have a small number of tasks
	# coverage = np.random.power(a,size=df.shape[0])*max_num
	# coverage = [max(1,max_num - c) for c in coverage.astype(int)]
	# print(coverage)
	# choice = [np.random.choice(max_num+1,size=c) for c in coverage]

	# fix number of model coverage
	coverage_list, choice = set_model_coverage(data,count,model_info,coverage,categories)
	# fix number of task coverage
	# coverage_list, choice = set_task_coverage(count,model_info,coverage,categories)
	print(coverage_list)

	df['#task'] = coverage_list
	df['tasks'] = [' '.join(c) for c in choice] #.astype(str)
	# return df
	df.to_csv(new_path)

if __name__ == '__main__':

	categories_coco = ['person','bicycle','car','motorcycle','airplane','bus','train',\
                'truck','boat','traffic_light','fire_hydrant','stop_sign','parking_meter',\
                'bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra',\
                'giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis',\
                'snowboard','sports_ball','kite','baseball_bat','baseball_glove','skateboard',\
                'surfboard','tennis_racket','bottle','wine_glass','cup','fork','knife','spoon',\
                'bowl','banana','apple','sandwich','orange','broccoli','carrot','hot_dog','pizza',\
                'donut','cake','chair','couch','potted_plant','bed','dining_table','toilet','tv',\
                'laptop','mouse','remote','keyboard','cell_phone','microwave','oven','toaster',\
                'sink','refrigerator','book','clock','vase','scissors','teddy_bear','hair_drier','toothbrush']

	categories_seg = ['airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
     	'car', 'cat', 'chair', 'cow', 'dining_table', 'dog', 'horse', 'motorcycle',
    	'person', 'potted_plant', 'sheep', 'couch', 'train', 'tv']

	# variants: hub_root
	model_info = {
		'yolov3':['ultralytics/yolov3','OD'],
		'yolov5x':['ultralytics/yolov5','OD'],
		'yolov5x6':['ultralytics/yolov5','OD'],
		'yolov5m6':['ultralytics/yolov5','OD'],
		'nvidia_ssd':['NVIDIA/DeepLearningExamples:torchhub','OD'],
		'fcn_resnet50':['pytorch/vision:v0.10.0','Seg'],
		'fcn_resnet101':['pytorch/vision:v0.10.0','Seg'],
		'deeplabv3_resnet50':['pytorch/vision:v0.10.0','Seg'],
		'deeplabv3_resnet101':['pytorch/vision:v0.10.0','Seg'],
		'deeplabv3_mobilenet_v3_large':['pytorch/vision:v0.10.0','Seg'],
		# 'fast_rcnn_R_50_FPN_1x':['COCO-Detection','detr2'],
    	'faster_rcnn_R_101_C4_3x':['COCO-Detection','detr2'],
    	'faster_rcnn_R_101_DC5_3x':['COCO-Detection','detr2'],
    	'faster_rcnn_R_101_FPN_3x':['COCO-Detection','detr2'],
		'faster_rcnn_R_50_C4_1x':['COCO-Detection','detr2'],
		'faster_rcnn_R_50_C4_3x':['COCO-Detection','detr2'],
		'faster_rcnn_R_50_DC5_1x':['COCO-Detection','detr2'],
		'faster_rcnn_R_50_DC5_3x':['COCO-Detection','detr2'],
		'faster_rcnn_R_50_FPN_1x':['COCO-Detection','detr2'],
		'faster_rcnn_R_50_FPN_3x':['COCO-Detection','detr2'],
		'faster_rcnn_X_101_32x8d_FPN_3x':['COCO-Detection','detr2'],
		'retinanet_R_101_FPN_3x':['COCO-Detection','detr2'],
		'retinanet_R_50_FPN_1x':['COCO-Detection','detr2'],
		'retinanet_R_50_FPN_3x':['COCO-Detection','detr2'],
		# 'rpn_R_50_C4_1x':['COCO-Detection','detr2'],
		# 'rpn_R_50_FPN_1x':['COCO-Detection','detr2'],
		'mask_rcnn_R_50_FPN_3x':['COCO-InstanceSegmentation','detr2'],
		'mask_rcnn_R_101_C4_3x':['COCO-InstanceSegmentation','detr2'],
		'mask_rcnn_R_101_DC5_3x':['COCO-InstanceSegmentation','detr2'],
		'mask_rcnn_R_101_FPN_3x':['COCO-InstanceSegmentation','detr2'],
		'mask_rcnn_R_50_C4_1x':['COCO-InstanceSegmentation','detr2'],
		'mask_rcnn_R_50_C4_3x':['COCO-InstanceSegmentation','detr2'],
		'mask_rcnn_R_50_DC5_1x':['COCO-InstanceSegmentation','detr2'],
		'mask_rcnn_R_50_DC5_3x':['COCO-InstanceSegmentation','detr2'],
		'mask_rcnn_R_50_FPN_1x':['COCO-InstanceSegmentation','detr2'],
		# 'mask_rcnn_R_50_FPN_1x_giou':['COCO-InstanceSegmentation','detr2'],
		'mask_rcnn_X_101_32x8d_FPN_3x':['COCO-InstanceSegmentation','detr2'],
		# 'Q2L-R101-448':['query2label','ML'],
		# 'Q2L-R101-576':['query2label','ML'],
		# 'Q2L-TResL-448':['query2label','ML'],
		# 'Q2L-TResL_22k-448':['query2label','ML'],
		# 'Q2L-SwinL-384':['query2label','ML'],
		# 'Q2L-CvT_w24-384':['query2label','ML']
	}

	# config
	path = 'config.yaml'
	with open(path) as f:
		config = yaml.load(f, Loader=yaml.FullLoader)

	# Parameters
	data = 'voc'
	change_task = True #True False
	coverage = 30
	# prob = '0.7'
	level = 'high'

	# Output path
	old_path = 'coco_model_config_new_model_30_'+level+'.csv' #'model_config_new_model_'+str(coverage)+'.csv' 
	if change_task:
		new_path = data+'_model_config_new_model_'+str(coverage)+'_'+level+'.csv' #old_path #'model_config_new_model_'+str(coverage)+'_prob_0.7.csv '_prob_0.7
	else:
		new_path = old_path
	# if os.path.isfile(new_path):
	# 	os.remove(new_path)


	# all: len(model_info)
	# newly_added: 27
	if not change_task:
		latency,prob = convert(len(model_info),config)
		write2file(model_info,latency,prob,old_path) # model_info,latency,prob,path='model_config.csv',modify=[]
	
	task_coverage = set_coverage(data,old_path,model_info,config,coverage=coverage,new_path=new_path,categories=[categories_coco,categories_seg])

	merge = False
	modify = [40,79]
	if merge:
		if os.path.isfile(old_path):
			df = pd.read_csv(old_path,index_col=0)
			new_df = pd.read_csv(new_path,index_col=0)
			frames = [df,new_df]
			results = pd.concat(frames,ignore_index=True)
			results['index'] = range(1,len(results)+1)
			results = results.drop(range(modify[0],modify[1]+1))
			results.index = range(len(results))
			results.to_csv(old_path)


	