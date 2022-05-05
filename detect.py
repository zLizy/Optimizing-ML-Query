import os
import argparse
import pandas as pd
import random
import re
import glob

from sympy.parsing.sympy_parser import parse_expr
import json, cv2, random
import matplotlib.pyplot as plt
# from google.colab.patches import cv2_imshow


import numpy as np
from random import choice
from PIL import Image
import torch
import time
import sys
sys.path.append('./test')
from my_logger import get_logger
sys.path.insert(0, './evalutation/models')

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
    if args.init or not os.path.isfile(path):
        df = pd.read_csv('output/tables/base_'+args.data_type+'_test.csv',index_col=0)
        return df, df.index, df['filename']
    else:
        df = pd.read_csv(path,index_col=0)
    df_copy = df.copy()

    print(args.conditions)
    # print(conditions)
    df = filter_data_with_condition(args.conditions,df)
    return df_copy,df.index,df['filename']
    
def filter_data_with_condition(conditions,df):
    conditions = conditions.split(',')
    if conditions == ['']:
        print('len(df)',len(df))
        return df
    for condition in conditions:
        df = filter_data_with_condition_layer(condition,df)
    # logger.info('~~~~~~~~~')
    # logger.info(df.head())
    # logger.info('~~~~~~~~~')
    return df

def filter_data_with_condition_layer(condition,df):
    if condition == ['']:
        print('len(df)',len(df))
        return df
    if 'or' in condition:
        tasks = [re.search(r'\w+', item).group() for item in condition.split(' or ')]
        if '=' in condition:
            if len(tasks) == 2:
                df = df[(df[tasks[0]].isnull()) | (df[tasks[1]].isnull())]
            if len(tasks) == 3:
                df = df[(df[tasks[0]].isnull()) | (df[tasks[1]].isnull()) | (df[tasks[2]].isnull())]
            print('len(df)',len(df))
            return df
        else:
            if len(tasks) == 2:
                df = df[(df[tasks[0]].notnull()) | (df[tasks[1]].notnull())]
            if len(tasks) == 3:
                df = df[(df[tasks[0]].notnull()) | (df[tasks[1]].notnull()) | (df[tasks[2]].notnull())]
            print('len(df)',len(df))
            return df
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
        # print(model_zoo.get_config_file(hub_root+'/'+ model_name+'.yaml'))
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

def compare_to_GT(index,df,df_model,model_idx,tasks,q_type,query_idx,bound,repository_file):
    # index,df,df_model,model_idx,tasks,args.q_type,args.query_idx,args.bound,args.repository_file

    # Ground truth
    GT_dir = '/home/zli/experiments/datasets/coco/data/annotations/'
    GT_file = GT_dir + 'val2017.csv'
    df_gt = pd.read_csv(GT_file,index_col=0)

    # Query
    filepath='simulation/coco_voc_query_'+q_type+'_gt.csv'
    df_query = pd.read_csv(filepath,index_col=0)
    df_query = df_query.iloc[query_idx]
    query = df_query['query']
    gt_index = df_query['index'].strip("[]").replace("'","").split(', ')
    # print(gt_index[:5])

    # Model repository
    df_repository = pd.read_csv(repository_file,index_col=0)

    df_base = pd.DataFrame()
    df_base['filename'] = df_gt['filename']
    predicates = [str(p) for p in list(parse_expr(query).atoms())]
    record = {}
    for t in tasks:
        if t in predicates:
            # gt_img = set.intersection(set(df_gt[df_gt[t].notnull()]['filename']),gt_index,set(df['filename']))

            inf_img = set(df[df[t].notnull()]['filename'])
            # accuracy = df_repository.loc['model_'+str(model_idx),t]
            record[t] = {'accuracy':round(accuracy,3),'inf':len(inf_img),'gt':len(gt_img),'TP[intersection]':len(set.intersection(gt_img,inf_img)),'FN[gt-inf]':len(gt_img-inf_img),'FP[inf-gt]':len(inf_img-gt_img)}
    
    
    for k, v in record.items():
        print(k,v)
    print()
            

def getLabels(df,model_idx,index,executable_filename,tasks,out_dir,data_type,assignment={},conditions={},latency=0,args=[],count=0): 
    #q_type='dnf',query_idx=0,bound=0,repository_file='',verify=False):
    # q_type=args.q_type,query_idx=args.query_idx,bound=args.bound,repository_file=args.repository_file,verify=args.verify
    # getLabels(result_df,model_idx,executable_index, tasks=tasks,latency=latency)
    addition = out_dir.split('_')[-1]
    if addition == 'all':
        addition = out_dir.split('_')[-2]

    logger.info('evaluation/inference_'+data_type+'_high/'+str(count)+'/summary_'+str(model_idx)+'_*.csv')
    path = glob.glob('evaluation/inference_'+data_type+'_high/'+str(count)+'/summary_'+str(model_idx)+'_*.csv')[0]
    numbers = re.findall( r'\d+', path, re.I)
    model_cost = int(numbers[-1])+latency
    ratio = len(index)/len(df)
    inference_cost = round( model_cost * ratio,2) #int(numbers[1])/(num_imgs/1000)

    logger.info('Executable df length: {}'.format(len(index)))
    logger.info('Executable ratio (selectivity): {}'.format(ratio))
    logger.info('Inference cost: {}'.format(inference_cost))
    logger.info('Original model cost: {}'.format(model_cost))
    
    # # evaluated inference cost
    # # Model repository
    # df_repository = pd.read_csv(args.repository_file,index_col=0)
    # model_cost = model_cost

    df_model = pd.read_csv(path,index_col=0)
    for task in tasks:
        if task != '__background__':
            df.loc[index,task] = df_model.loc[index,task]
    # time.sleep(latency*len(index))
    
    logger.info('model_{}'.format(model_idx))

    if args.verify:
        logger.info('task: {}'.format(assignment['model_'+str(model_idx)]))
        # compare_to_GT(index,df,df_model,model_idx,tasks,args.q_type,args.query_idx,args.bound,args.repository_file)
        record_GT(index,executable_filename,df,df_model,model_idx,assignment,conditions,out_dir,args.base,count)

    return df,inference_cost, model_cost, round(ratio,3)

def get_index(condition,appended_condition,df,df_model,df_gt,df_all,df_model_all,df_gt_all,task,df_record,out_dir,base,model_name,count,level,executable_filename,seperate=False):
    # intermediate step
    # inference
    if seperate:
        # seperate step
        df_ = filter_data_with_condition(condition,df_all)
        df_gt_ = filter_data_with_condition(condition,df_gt_all)
        df_model_ = df_model_all[df_model_all['filename'].isin(df_gt_['filename'])]
    else:
        df_ = filter_data_with_condition_layer(condition,df)
        # gt
        df_gt_ = filter_data_with_condition_layer(condition,df_gt)
        gt_index = df_gt_['filename']   
        df_model_ = df_model[df_model['filename'].isin(gt_index)]
        # model inference on gt filtered subset
        df = df_
        df_model = df_model_
        df_gt = df_gt_
        condition = appended_condition[:-1]

    gt_index = df_gt_['filename']
    index = df_.index
    total_len = len(df_model_.index)

    positive_model_index = df_model_[df_model_[task].notnull()]['filename']
    # true positive in filtered ground truth
    positive_gt_index = df_gt_[df_gt_[task].notnull()]['filename']
    # positive index in filtered model inference
    positive_exec_index = df_[df_[task].notnull()]['filename']
    # positive inference in filtered ground truth

    intersect_exec_model = set(positive_exec_index).intersection(set(positive_model_index))
    intersect_exec_gt = set(positive_exec_index).intersection(set(positive_gt_index))
    diff_exec_model = set(positive_model_index) - set(positive_exec_index)
    diff_exec_gt = set(positive_gt_index) - set(positive_exec_index)

    df_record = write_record(df_record,out_dir,base,model_name,task,count,level,seperate,
                condition,index,gt_index,total_len,
                intersect_exec_model,positive_exec_index,intersect_exec_gt,diff_exec_model,
                positive_model_index,diff_exec_gt,positive_gt_index,executable_filename)

    return df, df_gt, df_model, df_record

def write_record(df_record,out_dir,base,model_name,task,count,level,seperate,
                appended_condition,index,gt_index,total_len,
                intersect_exec_model,positive_exec_index,intersect_exec_gt,diff_exec_model,
                positive_model_index,diff_exec_gt,positive_gt_index,executable_filename):
    df_record = df_record.append({
                            'out_dir':out_dir,
                            'configuration':base,
                            'model_name':model_name,
                            'task':task,
                            'count':count,
                            'level':level,
                            'seperate':seperate,
                            'condition':appended_condition,
                            '#exec_img':len(index),
                            '#gt':len(gt_index),
                            'executable_ratio': round(len(index)/len(gt_index),3),
                            'selectivity':round(len(index)/total_len,3) if total_len != 0 else 0,
                            'exec_model_ratio':round(len(intersect_exec_model)/len(positive_exec_index),3),
                            'exec_gt_ratio':round(len(intersect_exec_gt)/len(positive_exec_index),3),
                            'missing_tp_exec_model_ratio':round(len(diff_exec_model)/len(positive_model_index),3) if len(positive_model_index) != 0 else 0,
                            'missing_tp_exec_gt_ratio':round(len(diff_exec_gt)/len(positive_gt_index),3),
                            '#missing_tp_exec_model':len(diff_exec_model),
                            '#missing_tp_exec_gt':len(diff_exec_gt),
                            '#intersect_exec_model':len(intersect_exec_model),
                            '#intersect_exec_gt':len(intersect_exec_gt),
                            '#positive_exec_index': len(positive_exec_index), 
                            '#positive_model_index': len(positive_model_index),
                            'positive_exec_index': positive_exec_index.values, 
                            'positive_model_index': positive_model_index.values,
                            'positive_gt_index':positive_gt_index.values,
                            'gt_img':gt_index.values,
                            'exec_img':executable_filename.values},ignore_index=True)
    return df_record


def record_GT(index,executable_filename,df,df_model,model_idx,assignment,conditions,out_dir,base,count):
    record_path = 'output/{}/order_intermediate_results.csv'.format(out_dir)
    if not os.path.exists(record_path):
        df_record = pd.DataFrame(columns=['out_dir','configuration','model_name','task','count','level','seperate','condition','#exec_img','#gt',
                                        'executable_ratio','selectivity','exec_model_ratio','exec_gt_ratio',
                                        'missing_tp_exec_model_ratio','missing_tp_exec_gt_ratio',
                                        '#missing_tp_exec_model','#missing_tp_exec_gt',
                                        '#intersect_exec_model','#intersect_exec_gt',
                                        '#positive_exec_index','#positive_model_index',
                                        'positive_exec_index','positive_model_index','positive_gt_index','gt_img','exec_img'])
    else:
        df_record = pd.read_csv(record_path,index_col=0)
    
    # Ground truth
    GT_dir = '/home/zli/experiments/datasets/coco/data/annotations/'
    GT_file = GT_dir + 'val2017.csv'
    df_gt = pd.read_csv(GT_file,index_col=0)
    df_gt = df_gt[df_gt['filename'].isin(df_model['filename'])]

    df_all =df.copy()
    df_gt_all = df_gt.copy()
    df_model_all = df_model.copy()

    logger.info(conditions)
    conditions = conditions.split(',')
    
    model_name = 'model_{}'.format(model_idx)
    for task in assignment[model_name]:
        logger.info('====')
        logger.info(task)

        level = get_level(','.join(conditions))
        logger.info('level: {}'.format(level))

        appended_condition = ''
        for condition in conditions:
            print('condition {}'.format(condition))
            appended_condition += condition + ','
            df, df_gt, df_model, df_record = get_index(condition,appended_condition,
                                                        df,df_model,df_gt,df_all,df_model_all,df_gt_all,task,df_record,
                                                        out_dir,base,model_name,count,level,executable_filename,seperate=False)
            df, df_gt, df_model, df_record = get_index(condition,appended_condition,
                                                        df,df_model,df_gt,df_all,df_model_all,df_gt_all,task,df_record,
                                                        out_dir,base,model_name,count,level,executable_filename,seperate=True)

    df_record.to_csv(record_path)

def get_level(conditions):
    conditions = conditions.split(',')
    # print(conditions)
    level = 0
    if conditions == ['']:
        # print('len(df)',len(df))
        return 0
    for condition in conditions:
        if 'or' in condition:
            tasks = [re.search(r'\w+', item).group() for item in condition.split(' or ')]
            level += len(tasks)
        else:
            level += 1
    return level

def inference(cat_dict,outdir,args):
    # preprocess start
    # addition = '_ap_dnf_accu_2'


    outdir = args.outdir
    config_file = args.config

    logger.info('\n========= {} ========'.format(args.base))    
    logger.info('outdir: {}'.format(outdir))
    logger.info('config_file:{}'.format(config_file))
    
    numbers = re.findall( r'\d+', args.base, re.I)
    query_idx = numbers[0]
    bound = numbers[2]
    
    logger.info('query_idx: {}'.format(query_idx))
    logger.info('bound: {}'.format(bound))

    obj_root = 'output/'+args.outdir+'/'
    if 'cost' in args.base:
        constraint = 'cost'
    else:
        constraint = 'accuracy'

    if 'order' in args.base:
        approach = 'order_opt'
        obj_file = obj_root+'summary_order_'+constraint+'_optimizer.csv'
    elif 'basic' in args.base:
        approach = 'basic_opt'
        obj_file = obj_root+'summary_'+constraint+'_optimizer.csv'
    else:
        approach = 'baseline'
        obj_file = obj_root+'summary_'+constraint+'_baseline_pareto.csv'

    ####### Objectives ##########
    df_obj = pd.read_csv(obj_file)
    if constraint == 'accuracy':
        row = df_obj[(df_obj['query_index']==int(query_idx)%5) & (df_obj['bound']==int(bound)/100)]
        obj = row['cost'].values
    else:
        row = df_obj[(df_obj['query_index']==int(query_idx)%5) & (df_obj['bound']==int(bound))]
        obj = row['accuracy'].values

    logger.info('~~~~~')
    logger.info('Objective: {}'.format(obj))
    logger.info('df_obj row:')
    logger.info(row)
    logger.info('~~~~~')

    if args.verify:
        plans = row['selected_model'].apply(eval).values[0]
        # switch keys and values
        assignment = dict()
        for k,v in plans[0].items():
            assignment.setdefault(v, []).append(k)
        
    if not os.path.exists('output/tables/'+outdir):
        os.makedirs('output/tables/'+outdir)

    # count = range(10)
    count = range(0,1)
    for i in count:
        logger.info('\n~~~~~~ count={} ~~~~~~'.format(i))
        preprocess_start = time.time()

        logger.info('approach: {}'.format(approach))

        path = 'output/tables/'+outdir+'/'+str(i)+'/'+args.base+'.csv'
        if not os.path.exists('output/tables/'+outdir+'/'+str(i)+'/'):
            os.makedirs('output/tables/'+outdir+'/'+str(i))

        time_path = 'output/time/'+outdir+'/'+str(i)+'/'+args.base+'.csv'
        if not os.path.exists('output/time/'+outdir+'/'+str(i)+'/'):
            os.makedirs('output/time/'+outdir+'/'+str(i))

        result_df,executable_index,executable_filename = get_executable_data(args,path)

        # Load time
        # path_time = 'output/tables/'+outdir+'/'+str(i) +'/time_'+approach+'.csv'
        if args.init or not os.path.isfile(time_path):
            df_time = pd.read_csv('output/tables/time.csv',index_col=0)
        else:
            df_time = pd.read_csv(time_path,index_col=0)

        #'convert/model_config_task.csv' # 'repository/model_config_task_half_all.csv'
        model_row = pd.read_csv(config_file, index_col='index').loc[args.experiment_index]
        
        # retrieve model information
        model_idx = args.experiment_index
        model_name = model_row['model']
        hub_root = model_row['root']
        model_type = model_row['type']
        model_info = [model_name,hub_root,model_type]

        if model_type == 'Seg':
            categories = cat_dict['Seg']
        else:
            categories = cat_dict['Others']

        latency = int(model_row['latency'])
        percentage = model_row['probability']
        tasks = model_row['tasks'].strip('][').split(' ') 
        # tasks = [categories[i] for i in tasks if i<len(categories)]
        
        logger.info('model index: {}'.format(model_idx))
        logger.info('latency: {}'.format(latency))

        # preprocess end
        try:
            preprocess_time = int((time.time() - preprocess_start)/(len(executable_index)/1000))
        except:
            preprocess_time = 0

        # logger.info('assignment: {}'.format(assignment))
        # print(result_df.head())
        if args.verify:
            result_df, _, _, _ = getLabels(result_df,model_idx,executable_index,executable_filename, tasks,outdir,args.data_type,assignment,args.conditions,latency=latency,args=args,count=i)
        else:
            result_df, inference_time, model_cost, ratio = getLabels(result_df,model_idx,executable_index,executable_filename, tasks,outdir,args.data_type,latency=latency,args=args,count=i)

        # logger.info('~~~~ result_df ~~~~~~~')
        # logger.info(result_df.head())
        # logger.info('~~~~~~~~~~~')

        # model_time = int((time.time() - model_start)/(len(df)/1000))
        result_df.to_csv(path)

        if not args.verify:
            total_time = int(inference_time+preprocess_time)

            logger.info('preprocess time: {}'.format(preprocess_time))
            logger.info('inference time: {}'.format(inference_time))
            logger.info('total time: {}'.format(total_time))

            row_time = pd.DataFrame([{'scriptname':args.base,'model':args.experiment_index, \
                            'conditions':args.conditions, 'preprocess_time':preprocess_time, \
                            'model_inference':inference_time,'total':total_time, 'model_cost':model_cost}])
            df_time = df_time.append(row_time, ignore_index=True)
            df_time.to_csv(time_path)

            logger.info('~~~~')
            logger.info('df_time for the inference:')
            logger.info(df_time)
            logger.info('~~~~')
            
            logger.info('Current step:')
            logger.info('model: '+ str(model_idx)+', cost: '+str(model_cost)+', ratio: '+str(ratio)+ ', preprocess: '+str(preprocess_time)+ ', model inference: '+ str(inference_time))
            
            logger.info('Total time: {}'.format(df_time['total'].sum()))
            logger.info('Objective: {}'.format(obj))
            # logger.info('#############################')


def build_argument_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--experiment_index", type=int, required=True)
    parser.add_argument("--conditions",type=str,default='')
    parser.add_argument("--base", type=str,required=True)
    parser.add_argument("--outdir", type=str)
    parser.add_argument("--config",type=str,default='model_config_new_model_20.csv')
    parser.add_argument("--data_type",type=str,default='coco')

    parser.add_argument("--verify", action='store_true')
    parser.add_argument("--q_type", type=str)
    parser.add_argument("--query_idx",type=int)
    parser.add_argument("--bound", type=int)
    parser.add_argument("--repository_file",type=str)
    parser.add_argument("-init","--init",action='store_true',help="First time executing the script")
    return parser


if __name__ == "__main__":
    '''
    Command:
    python3 detect.py --experiment_index 1 --conditions person>1,car=0
    '''

    parser = build_argument_parser()
    args, _ = parser.parse_known_args()
    
    # args.verify = True

    if args.verify:
        logger = get_logger(__name__, 'logs/verify_accuracy_cnf_cost_precision.log', use_formatter=False)
    else:
        logger = get_logger(__name__, 'logs/execution_voc_pareto_accuracy.log', use_formatter=False)

    outdir = 'output'

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

    logger.info('categories_coco:')
    logger.info(categories_coco)


    categories_seg = ['__background__', 'airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
     'car', 'cat', 'chair', 'cow', 'dining_table', 'dog', 'horse', 'motorcycle',
    'person', 'pottedplant', 'sheep', 'couch', 'train', 'tv']

    logger.info('categories_seg:')
    logger.info(categories_seg)

    
    cat_dict = {'Seg':categories_seg,'Others':categories_coco}


    inference(cat_dict,outdir,args)
