import pandas as pd
import os

# categories = ['person','chair','car','dining_table','cup','bcategoriesottle','bowl']
categories = ['toaster', 'hair_drier', 'scissors', 'toothbrush', 'parking_meter', 'bear', 'snowboard', \
        'hot_dog', 'microwave', 'donut', 'sheep', 'stop_sign', 'broccoli', 'apple', 'carrot', 'frisbee', \
        'orange', 'zebra', 'fire_hydrant', 'cow', 'mouse', 'elephant', 'kite', 'teddy_bear', 'airplane', \
        'baseball_bat', 'sandwich', 'baseball_glove', 'giraffe', 'refrigerator', 'banana', 'suitcase', \
        'keyboard', 'wine_glass', 'oven', 'skis', 'boat', 'cake', 'bird', 'skateboard', 'horse', 'vase', \
        'remote', 'tie', 'bicycle', 'toilet', 'bed', 'surfboard', 'spoon', 'pizza', 'fork', 'train', \
        'motorcycle', 'tennis_racket', 'sports_ball', 'potted_plant', 'umbrella', 'dog', 'knife', 'laptop', \
        'cat', 'sink', 'bus', 'traffic_light', 'couch', 'clock', 'tv', 'cell_phone', 'backpack', 'book', \
        'bench', 'truck', 'handbag', 'bowl', 'bottle', 'cup', 'dining_table', 'car', 'chair', 'person']

type = 'val'
image_path = '/home/zli/fiftyone/voc_2012_'+type+'/data'
imgs = os.listdir(image_path)

## COCO
# imgs_val = os.listdir('/home/zli/experiments/datasets/coco/images/val')

df = pd.DataFrame(columns=categories)
df['filename'] = imgs
# df['filename'] = imgs_val+imgs_test

# df = df.set_index('filename')

df.to_csv('base_voc_'+type+'.csv')

