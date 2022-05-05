import os
import argparse
import pandas as pd
import random
import re
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
    parser.add_argument("--experiment_index", type=int, required=True)
    parser.add_argument("--conditions",type=str,default='')
    parser.add_argument("--base", type=str,required=True)
    parser.add_argument("--outdir", type=str)
    return parser

# Inference
def detect(model, image_path, tasks, latency, percentage, image_dir='',count=0, names=[]):

    # Images
    img = Image.open(image_dir+image_path)

    prediction = model(img, size=640)  # includes NMS'
    pred = prediction.pred[0]
    img = prediction.imgs[0]
    time.sleep(latency)

    ans = {}
    pred = prediction.pandas().xyxy[0]
    # print(pred)
    if pred is not None:
        for i,row in pred.iterrows():
            n = row['name']
            if np.random.rand() > percentage:
                continue
            if n in names:
                if n not in ans.keys():
                    ans[n] = round(row['confidence'],2)
                else:
                    ans[n] = max(round(row['confidence'],2), ans[n])

    return ans

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
    

def inference():
    # preprocess start
    # addition = '_ap_dnf_accu_2'

    parser = build_argument_parser()
    args, _ = parser.parse_known_args()

    outdir = args.outdir

    if not os.path.exists('output/tables/'+outdir):
        os.makedirs('output/tables/'+outdir)
        
    preprocess_start = time.time()

    categories = ['person','chair','car','cup','bottle','bowl','tv']

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
    latency = 0
    # lag = 1e6
    tasks = range(80)
    # image_dir = '/home/zli/experiments/datasets/coco/test/'
    image_dir = '/home/zli/experiments/Project_Boolean_simple/datasets/coco/images/test/'
    
    inference_map = {0:'yolov3', 1:'yolov5',2:'yolov5'}

    config_file = 'convert/model_config_task.csv' # 'repository/model_config_task_half_all.csv'
    query_parameters = pd.read_csv(config_file, index_col='index').loc[(args.experiment_index)]
    # dataset = query_parameters.dataset
    model_name = query_parameters.model
    latency = int(query_parameters.latency)/1000
    percentage = query_parameters.mis_prediction_prop
    tasks = [int(t) for t in query_parameters.tasks.strip('][').split(' ')]

    print('model index: ' + str(args.experiment_index))
    print('latency: '+str(latency))

    # Model
    structure = inference_map[(args.experiment_index-1)//50]
    model = torch.hub.load('ultralytics/'+structure, model_name, pretrained=True,
                           force_reload=False).autoshape()  # for PIL/cv2/np inputs and NMS

    # try:
    #     model = model.to(torch.device('cuda:0'))
    #     print(torch.cuda.current_device())
    # except:
    #     model = model.to(torch.device('cuda:1'))
    #     print(torch.cuda.current_device())

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = torch.nn.DataParallel(model)
    #     model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # preprocess end
    preprocess_time = int((time.time() - preprocess_start)/(len(df)/1000))
    print('preprocess: '+str(preprocess_time))

    # start model inference
    model_start = time.time()
    for row in result_df.itertuples():
        detected_objects = detect(model, row.filename, tasks=tasks,
                                         latency=latency, percentage=percentage, names=categories,image_dir=image_dir)

        for k, v in detected_objects.items():
            df.loc[row.Index, k] = v
    model_time = int((time.time() - model_start)/(len(df)/1000))

    # check evaluated cost vs. inference cost
    df_model = pd.read_csv('repository/model_stats_ap.csv',index_col=0)
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

    inference()
