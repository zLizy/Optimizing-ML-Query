import re
import glob
import pandas as pd
from pathlib import Path

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', \
			'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', \
			'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', \
			'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', \
			'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', \
			'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', \
			'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', \
			'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', \
			'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', \
			'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'toothbrush']



# get task coverage
df_config = pd.read_csv('model_config_task.csv')


model_repository = {}
for path in glob.glob('../output/summary/*.txt'):
	with open(path,'r') as file:
		model = Path(path).stem
		if model == 'model_-1' or model == 'model_0':
			continue
		content = file.readlines()

		idx = int(re.search( r'\d+', model, re.I).group())
		print(idx)
		if idx <= 100:
			base = 6.9 #1530 # 6.9
		else:
			base = 2.3 #165 # 2.3

		task_cov = df_config.loc[df_config['index']==idx]['tasks'].values[0]#[1:-1]
		task_cov = task_cov.replace('\n','').split(' ')
		task_cov = [int(cov) for cov in task_cov if cov != '']

		# extract runtime
		inference = float(content[2].split(':')[1].replace(' ','').replace('\n',''))
		runtime = base * 16 #(int(inference) - base) * 15 + int(inference)
		# print('runtime',runtime)

		
		class_list = content[8:]
		# print('len',len(class_list))
		
		# print(task_cov)
		# print('len',len(names))
		# for i,task in enumerate(task_cov):
		# 	print(task)
		# 	print(names[task])
		# 	print(class_list[task].split('\t')[3])
		model_repository[model] = {names[task]:float(class_list[task].split('\t')[3]) for i, task in enumerate(task_cov)}
		model_repository[model]['cost'] = runtime

df = pd.DataFrame.from_dict(model_repository).T
print(df.head())
print(df['cost'])
print(df.shape)
df.to_csv('model_stats_new.csv')