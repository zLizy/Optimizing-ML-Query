import os
import argparse
import pandas as pd
import random
import re

import detectron2
import json, cv2, random
import matplotlib.pyplot as plt
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
# from run_io.db_adapter import convertDSNToRfc1738
# from run_io.extract_table_names import extract_tables
# from sqlalchemy import create_engine

import numpy as np
from random import choice
from PIL import Image
import torch
import time
import sys

sys.path.insert(0, './test/models')

def build_argument_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--experiment_index", type=int, required=True)
    parser.add_argument("--conditions",type=str,default='')
    parser.add_argument("--base", type=str,required=True)
    parser.add_argument("--outdir", type=str)
    return parser

def yolo(img_path, model,tasks,latency, percentage,categories=[],inference=True):
    import test_yolo
    return test_yolo.run(img_path, model,tasks,latency, percentage,categories,inference)
    pass

def detr(img_path, model,tasks,latency, percentage,categories,inference=True):
    import test_detr2
    return test_detr2.inference(img_path, model,tasks,latency, percentage,categories,inference=True)
    pass

def ssd(img_path, model,tasks,latency, percentage,categories,inference=True):
    import test_ssd
    return test_ssd.inference(img_path,model, tasks,latency, percentage,categories,inference=True)
    pass

def segment(img_path,model, tasks,latency, percentage,categories,inference=True):
    # if 'fcn' in model_name:
    #     import test_fcn
    # else:
    #     import test_deeplabv3
    import test_fcn
    return test_fcn.inference(img_path,model, tasks,latency, percentage,categories,inference=True)
    pass

def multi_label_detect(model_info):
    pass

# Inference
def detect(image_path, model,model_info, tasks, latency, percentage, categories=[]):

    # Model
    model_name,hub_root,model_type = model_info

    function_dict = {
        'ultralytics/yolov3':yolo,
        'ultralytics/yolov5':yolo,
        'COCO-Detection':detr,
        'COCO-InstanceSegmentation':detr,
        'NVIDIA/DeepLearningExamples:torchhub':ssd,
        'pytorch/vision:v0.10.0':segment,
        'query2label':multi_label_detect
    }

    try: 
        labels = function_dict[hub_root](image_path, model, tasks=tasks, latency=latency, percentage=percentage,categories=categories,inference=True) 
    except KeyError: 
        print('Fail, no such model') 
    
    return labels

def get_executable_data(args,path):
    # args.base
    # args.conditions
    if not os.path.isfile(path):
        return pd.read_csv('output/tables/base.csv',index_col=0)
    else:
        df = pd.read_csv(path,index_col=0)
    
    print(args.conditions)
    conditions = args.conditions.split(',')
    print(conditions)

    if conditions == ['']:
        print('len(df)',len(df))
        return df
    for condition in conditions:
        if 'or' in condition:
            tasks = [re.search(r'\w+', item).group() for item in condition.split(' or ')]
            if '=' in condition:
                if len(tasks) == 2:
                    df = df[(df[tasks[0]].isnull()) | (df[tasks[1]].isnull())]
                if len(tasks) == 3:
                    df = df[(df[tasks[0]].isnull()) | (df[tasks[1]].isnull()) | (df[tasks[2]].isnull())]
                print('len(df)',len(df))
                continue
            else:
                if len(tasks) == 2:
                    df = df[(df[tasks[0]].notnull()) | (df[tasks[1]].notnull())]
                if len(tasks) == 3:
                    df = df[(df[tasks[0]].notnull()) | (df[tasks[1]].notnull()) | (df[tasks[2]].notnull())]
                print('len(df)',len(df))
                continue
        if '=' in condition:
            task = condition.split('=')[0]
            df = df[df[task].isnull()]
            print('len(df)',len(df))
        if '>' in condition:
            task = condition.split('>')[0]
            df = df[df[task].notnull()]
            print('len(df)',len(df))
    return df
    
def getModel(model_info):
    model_name,hub_root,model_type = model_info

    if model_type == 'detr2':
    
        cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file(hub_root+'/'+ model_name+'.yaml'))
        print(model_zoo.get_config_file(hub_root+'/'+ model_name+'.yaml'))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(hub_root+'/'+ model_name+'.yaml')

        predictor = DefaultPredictor(cfg)
        return predictor
    else:
        try:
            model = torch.hub.load(hub_root, model_name, pretrained=True)#.autoshape()
            # model = torch.hub.load('ultralytics/'+structure, model_name, pretrained=True,
            #                    force_reload=False).autoshape()  # for PIL/cv2/np inputs and NMS
        except:
            model = torch.hub.load(hub_root, model_name, pretrained=True,force_reload=True).autoshape()  # for PIL/cv2/np inputs and NMS
        model.eval()
        if model_name == 'nvidia_ssd':
            utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
            return [model,utils]
        return model

def inference(categories):
    # preprocess start
    # addition = '_ap_dnf_accu_2'


    parser = build_argument_parser()
    args, _ = parser.parse_known_args()

    outdir = args.outdir

    if not os.path.exists('output/tables/'+outdir):
        os.makedirs('output/tables/'+outdir)
        
    preprocess_start = time.time()

    path = 'output/tables/'+outdir+'/'+args.base+'.csv'
    result_df = get_executable_data(args,path)

    if 'order' in args.base:
        approach = 'order_opt'
    elif 'basic' in args.base:
        approach = 'basic_opt'
    else:
        approach = 'baseline'


    if not os.path.isfile(path):
        df = pd.read_csv('output/tables/base.csv',index_col=0)
    else:
        df = pd.read_csv(path,index_col=0)
  
    # Load time
    path_time = 'output/tables/'+outdir+'/time_'+approach+'.csv'
    if not os.path.isfile(path_time):
        df_time = pd.read_csv('output/tables/time.csv',index_col=0)
    else:
        df_time = pd.read_csv(path_time,index_col=0)

    # Load model parameters
    dataset = 'coco_val'
    image_dir = '/home/zli/experiments/datasets/coco/images/test/'
    # image_dir = '/home/zli/experiments/Project_Boolean_simple/datasets/coco/images/test/'
    
    # inference_map = {0:'yolov3', 1:'yolov5',2:'yolov5'}

    config_file = 'convert/model_config_new_model_20.csv' #'convert/model_config_task.csv' # 'repository/model_config_task_half_all.csv'
    query_parameters = pd.read_csv(config_file, index_col='index').loc[(args.experiment_index)]
    
    # retrieve model information
    model_idx = args.experiment_index
    model_row = query_parameters
    model_name = model_row['model']
    hub_root = model_row['root']
    model_type = model_row['type']
    model_info = [model_name,hub_root,model_type]

    if model_type == 'Seg':
        categories = categories['Seg']
    else:
        categories = categories['Others']
    
    latency = int(model_row['latency'])/1000
    percentage = model_row['probability']
    tasks = [int(t) for t in model_row['tasks'].strip('][').split(' ')]
    tasks = [categories[i] for i in tasks if i<len(categories)]

    print('model index: ' + str(model_idx))
    print('latency: '+str(latency))

    
    # Model
    model = getModel(model_info)

    # preprocess end
    preprocess_time = int((time.time() - preprocess_start)/(len(df)/1000))
    print('preprocess: '+str(preprocess_time))

    # start model inference
    model_start = time.time()
    for row in result_df.itertuples():
        detected_objects = detect(image_dir+row.filename, model,model_info, tasks=tasks,latency=latency, percentage=percentage, categories=categories)
        # print(detected_objects)
        for k, v in detected_objects.items():
            df.loc[row.Index, k] = v
    model_time = int((time.time() - model_start)/(len(df)/1000))

    # check evaluated cost vs. inference cost
    df_model = pd.read_csv('repository/model_stats_f1_new_model_20.csv',index_col=0)
    model_name = 'model_'+str(args.experiment_index)
    print('cost: '+ str(df_model.loc[model_name,'cost']) +' '+'model inference: '+ str(model_time))
    
    df.to_csv(path)
    
    total_time = int(model_time+preprocess_time)

    df_time = df_time.append(pd.DataFrame([{'scriptname':args.base,'model':args.experiment_index,
                    'conditions':args.conditions, 'preprocess_time':preprocess_time,
                    'model_inference':model_time,'total':total_time, 'model_cost':df_model.loc[model_name,'cost']}]), ignore_index=True)
    df_time.to_csv('output/tables/'+outdir+'/time_'+approach+'.csv')
    
    print('preprocess: '+str(preprocess_time)+ ', model inference: '+ str(model_time))
    print('#############################')


if __name__ == "__main__":
    '''
    Command:
    python3 detect.py --experiment_index 1 --conditions person>1,car=0
    '''

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
    categories_coco = [c.replace(' ','_') for c in categories_coco if c!='N/A']

    categories_seg = ['__background__', 'airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
     'car', 'cat', 'chair', 'cow', 'dining_table', 'dog', 'horse', 'motorcycle',
    'person', 'pottedplant', 'sheep', 'couch', 'train', 'tv']

    
    cat_dict = {'Seg':categories_seg,'Others':categories_coco}

    inference(cat_dict)
