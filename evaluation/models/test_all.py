import os
import argparse
import pandas as pd
import random
# from run_io.db_adapter import convertDSNToRfc1738
# from run_io.extract_table_names import extract_tables
# from sqlalchemy import create_engine

import numpy as np
from random import choice
from PIL import Image
import torch
import time

def build_argument_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    # parser.add_argument("--experiment_index", type=int, required=True)
    # parser.add_argument("--conditions",type=str,default='')
    # parser.add_argument("--base", type=str,required=True)
    return parser

def yolo(img_path, model,tasks,latency, percentage,categories=[]):
    import models.test_yolo.run
    return run(img_path, model,tasks,latency, percentage,categories)
    pass

def detr(img_path, model,tasks,latency, percentage,categories):
    import models.test_detr.inference
    return inference(img_path, model,tasks,latency, percentage,categories)
    pass

def ssd(img_path, model,tasks,latency, percentage,categories):
    import models.test_ssd.inference
    return inference(img_path,model, tasks,latency, percentage,categories)
    pass

def segment(img_path,model, tasks,latency, percentage,categories):
    if 'fcn' in model_name:
        import models.test_fcn.inference
    else:
        import models.test_deeplabv3.inference
    return inference(img_path,model, tasks,latency, percentage,categories)
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
        'facebookresearch/detr':detr,
        'NVIDIA/DeepLearningExamples:torchhub':ssd,
        'pytorch/vision:v0.10.0':segment,
        'query2label':multi_label_detect
    }

    try: 
        labels = function_dict[hub_root](image_path, model_info, tasks=tasks, latency=latency, percentage=percentage,categories=categories) 
    except KeyError: 
        print('Fail, no such model') 
    
    return labels

def get_data(args,path,categories):
    # args.base
    # args.conditions
    if not os.path.isfile(path):
        return pd.read_csv('base.csv',index_col=0)
    else:
        df = pd.read_csv(path,index_col=0)
    conditions = args.conditions.split(',')
    # print(conditions)
    if conditions == ['']:
        return df
    for c in conditions:
        if '=' in c:
            task = c.split('=')[0]
            df_sub = df[df[task]==0]
        else:
            task = c.split('>')[0]
            df_sub = df[df[task]!=0]
    return df_sub
    
def getModel(model_info):
    model_name,hub_root,model_type = model_info
    try:
        model = torch.hub.load(hub_root, model_name, pretrained=True)#.autoshape()
        # model = torch.hub.load('ultralytics/'+structure, model_name, pretrained=True,
        #                    force_reload=False).autoshape()  # for PIL/cv2/np inputs and NMS
    except:
        model = torch.hub.load(hub_root, model_name, pretrained=True,force_reload=True).autoshape()  # for PIL/cv2/np inputs and NMS
    model.eval()
    return model

def inference(folder,test=False):

    # parser = build_argument_parser()
    # args, _ = parser.parse_known_args()

    # categories = ['person','chair','car','dining_table','cup','bottle','bowl']
    categories_coco = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
                        'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant', 
                        'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
                        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
                        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 
                        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports_ball', 
                        'kite', 'baseball_bat', 'baseball_glove', 'skateboard', 
                        'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 
                        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 
                        'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 
                        'cake', 'chair', 'couch', 'potted_plant', 'bed', 'dining_table', 
                        'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
                        'cell_phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
                        'book', 'clock', 'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush']

    categories_seg = ['__background__', 'airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
     'car', 'cat', 'chair', 'cow', 'dining_table', 'dog', 'horse', 'motorcycle',
    'person', 'potted_plant', 'sheep', 'sofa', 'train', 'tv']
  
    # Load model parameters
    dataset = 'coco_val'
    latency = 0
    lag = 1e6
    tasks = range(80)
    image_dir = '/home/zli/experiments/datasets/coco/images/val'
    if test:
        image_dir = '/home/zli/experiments/datasets/coco/images/test'
    # image_dir = '/home/zli/experiments/datasets/coco/val/'
    
    # inference_map = {0:'yolov3', 1:'yolov5',2:'yolov5'}

    query_parameters = pd.read_csv('../../convert/model_config_new_model_20.csv', index_col=0)


    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = torch.nn.DataParallel(model)
    #     model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # preprocess end
    # preprocess_time = time.time() - preprocess_start
    # print('preprocess: '+str(preprocess_time))

    path = '../base.csv'
    result_df = pd.read_csv(path,index_col=0)#get_data(path)
    for i,param_row in query_parameters.iloc[:140].iterrows():
        print('========')
        print(i,param_row['index'])
        print('========')
        start = time.time()

        df = pd.read_csv(path,index_col=0)#get_data(path)

        latency = param_row['latency']/1000
        percentage = param_row['probability']
        tasks = param_row['tasks']
        tasks = [categories_coco[int(t)] for t in tasks.strip('][').split(' ')]
        index = param_row['index']
        # structure = inference_map[(index-1)//50]
        
        model_name = param_row['model']
        hub_root = param_row['root']
        model_type = param_row['type']
        model_info = [model_name,hub_root,model_type]

        if model_type == 'Seg':
            categories = categories_seg
        else:
            categories = categories_coco

        model = getModel(model_info)

        # start model inference
        model_start = time.time()
        for row in result_df.itertuples():
            detected_objects = detect(image_dir+row.filename, model,model_info, tasks=tasks,latency=latency, percentage=percentage, categories=categories)

            for k, v in detected_objects.items():
                df.loc[row.Index, k] = v
        model_time = int((time.time() - model_start)/(len(df)/1000))  
    
        # print(df.head())
        df.to_csv('tables/summary_'+str(index)+'_'+str(int(model_time))+'.csv')

        print('model inference: '+ str(model_time))


if __name__ == "__main__":
    '''
    Command:
    python3 detect.py --experiment_index 1 --conditions person>1,car=0
    '''

    folder = 'inference'
    test = True
    inference(folder,test=test)
