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

def getModel(model_info):
    model_name,hub_root,model_type = model_info
    try:
        model = torch.hub.load(hub_root, model_name, pretrained=True)#.autoshape()
        # model = torch.hub.load('ultralytics/'+structure, model_name, pretrained=True,
        #                    force_reload=False).autoshape()  # for PIL/cv2/np inputs and NMS
    except:
        model = torch.hub.load(hub_root, model_name, pretrained=True,force_reload=True)#.autoshape()  # for PIL/cv2/np inputs and NMS
    model.eval()
    return model

# Inference
def run(image_path,model, tasks, latency, percentage,count=0, categories=[],inference=False):

    # Images
    img = Image.open(image_path)

    prediction = model(img, size=640)  # includes NMS'
    pred = prediction.pred[0]
    img = prediction.imgs[0]
    
    # time.sleep(latency)

    ans = {}
    pred = prediction.pandas().xyxy[0]
    # print(pred)
    if pred is not None:
        # if inference and n in tasks:
        for i,row in pred.iterrows():
            n = row['name'].replace(' ','_')
            # if n not in tasks:
            #     continue
            # if np.random.rand() > percentage:
            #     continue
            if n not in ans.keys():
                ans[n] = round(row['confidence'],2)
            else:
                ans[n] = max(round(row['confidence'],2), ans[n])
        
        # # gn = torch.tensor(img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        # latency = latency/1000
        # time.sleep(latency)

        # # Save results into files in the format of: class_index x y w h
        # for *xyxy, conf, cls in reversed(pred):
        #     cls = int(cls.item())
        #     if np.random.rand() > percentage:
        #         # cls = names.index(choice(names))
        #         continue
        #     if cls in tasks:
        #         if names[cls] not in ans.keys():
        #             ans[names[cls]] = conf.item()
        #         else:
        #             ans[names[cls]] = max(conf.item(), ans[names[cls]])
    # print(ans)
    return ans

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
    

def inference():
    # preprocess start
    preprocess_start = time.time()

    parser = build_argument_parser()
    args, _ = parser.parse_known_args()

    # categories = ['person','chair','car','dining_table','cup','bottle','bowl']
    # categories = ['toaster', 'hair_drier', 'scissors', 'toothbrush', 'parking_meter', 'bear', 'snowboard', \
    #     'hot_dog', 'microwave', 'donut', 'sheep', 'stop_sign', 'broccoli', 'apple', 'carrot', 'frisbee', \
    #     'orange', 'zebra', 'fire_hydrant', 'cow', 'mouse', 'elephant', 'kite', 'teddy_bear', 'airplane', \
    #     'baseball_bat', 'sandwich', 'baseball_glove', 'giraffe', 'refrigerator', 'banana', 'suitcase', \
    #     'keyboard', 'wine_glass', 'oven', 'skis', 'boat', 'cake', 'bird', 'skateboard', 'horse', 'vase', \
    #     'remote', 'tie', 'bicycle', 'toilet', 'bed', 'surfboard', 'spoon', 'pizza', 'fork', 'train', \
    #     'motorcycle', 'tennis_racket', 'sports_ball', 'potted_plant', 'umbrella', 'dog', 'knife', 'laptop', \
    #     'cat', 'sink', 'bus', 'traffic_light', 'couch', 'clock', 'tv', 'cell_phone', 'backpack', 'book', \
    #     'bench', 'truck', 'handbag', 'bowl', 'bottle', 'cup', 'dining_table', 'car', 'chair', 'person']
  
    # Load model parameters
    dataset = 'coco_val'
    latency = 0
    lag = 1e6
    tasks = range(80)
    image_dir = '/home/zli/experiments/datasets/coco/images/val/'
    inference_map = {0:'yolov3', 1:'yolov5',2:'yolov5'}

    query_parameters = pd.read_csv('../../convert/model_config_new_model_20.csv', index_col=0)


    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = torch.nn.DataParallel(model)
    #     model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # preprocess end
    preprocess_time = time.time() - preprocess_start
    print('preprocess: '+str(preprocess_time))

    path = '../base.csv'
    result_df = pd.read_csv(path,index_col=0)#get_data(path)
    for i,param_row in query_parameters.iloc[25:].iterrows():
        print('========')
        print(i,param_row['index'])
        print('========')
        start = time.time()
        
        df = pd.read_csv(path,index_col=0)#get_data(path)

        latency = param_row['latency']/1000
        percentage = param_row['probability']
        tasks = param_row['tasks']
        index = param_row['index']

        model_name = param_row['model']
        hub_root = param_row['root']
        model_type = param_row['type']
        model_info = [model_name,hub_root,model_type]

        model = getModel(model_info)

        # start model inference
        model_start = time.time()
        pre_time = model_start - start
        print('pre_time',pre_time)

        for row in result_df.itertuples():
            detected_objects = run(image_dir+row.filename, model, tasks=tasks,
                                            latency=latency, percentage=percentage)

            for k, v in detected_objects.items():
                df.loc[row.Index, k] = v
        model_time = int((time.time() - model_start)/(len(df)/1000))
    
        # print(df.head())
        df.to_csv('tables/summary_'+str(index)+'_'+str(int(model_time))+'.csv')

        print('preprocess: '+str(preprocess_time)+ ', model inference: '+ str(model_time))


if __name__ == "__main__":
    '''
    Command:
    python3 detect.py --experiment_index 1 --conditions person>1,car=0
    '''

    inference()
