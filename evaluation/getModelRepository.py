import os
import re
import glob
import pandas as pd
from pathlib import Path
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('../test')
from my_logger import get_logger

import util

def write_files(model_repository,out_file):
    df = pd.DataFrame.from_dict(model_repository).T
    df.to_csv(out_file)
    # df['dataset'] = 'coco_val'
    # df['image_dir'] = '/datasets/coco/images/val/val2017'
    # print(df.head())
    # print(df.shape)
    

def get_metrics(model_repository,model,num_run_dir,stem,metric_dir,task_cov,metric,selectivity=False):

    task_performance_dict = {}
    for run_dir in num_run_dir:
        path = metric_dir+'/'+run_dir+'/'+stem+'.csv'
        # runtime with one GPU
        # inference_map = {0:25.6, 1:16.4, 2:32.8, 3:39, 4:24.2, 5:27.4}
        df_metrics = pd.read_csv(path,index_col=0)
        df_metrics = df_metrics.set_index('class')
            
        for task in task_cov:
            try:
                row = df_metrics.loc[task]
            except:
                continue
            if metric == 'ap':
                task_performance_dict[task] = task_performance_dict.get(task,0) + round(row[metric.upper()],3)#*percentage,3)
            else:
                task_performance_dict[task] = task_performance_dict.get(task,0)+ round(row[metric],3)
                        
    for task in task_cov:
        try:
            task_performance_dict[task] = round(task_performance_dict[task]/len(num_run_dir),3)
        except:
            continue
    model_repository[model] = task_performance_dict #{names[task]:float(class_list[task].split('\t')[3]) }
    return model_repository

def getMR(path,dataset,num_task_coverage,metric,metric_dir,selectivity_dir,file_addition,all=False):
    # path,task_coverage,metric,metric_dir,selectivity_dir,file_addition

    logger.info('===============')
    logger.info('path: {}'.format(path))
    
    df_config = pd.read_csv(path,index_col=0)

    logger.info('------------')
    logger.info('df_config head')
    logger.info('\t'+ df_config.head().to_string().replace('\n', '\n\t')) 
    logger.info('------------')

    logger.info('dataset: {}'.format(dataset))
    logger.info('num_task_coverage: {}'.format(num_task_coverage))
    logger.info('metric: {}'.format(metric))
    logger.info('metric_dir: {}'.format(metric_dir))
    logger.info('selectivity_dir: {}'.format(selectivity_dir))
    logger.info('file_addition: {}'.format(file_addition))

    
    sel_model_repository = {}
    accu_model_repository = {}

    num_run_dir = os.listdir(metric_dir+'/')
    for i,path in enumerate(glob.glob(metric_dir+'/'+num_run_dir[0]+'/*.csv')):
        stem = Path(path).stem       
        numbers = re.findall( r'\d+', stem, re.I)
        idx = int(numbers[0])
        print('idx',idx)
        base = int(numbers[1]) #int(numbers[1])/(num_imgs/1000)
        model = 'model_'+numbers[0]

        logger.info('----------')
        logger.info('path: {}'.format(path))
        logger.info('model idx: {}'.format(idx))
        logger.info('model base cost: {}'.format(base))

        row = df_config.loc[df_config['index']==idx]

        logger.info('df_config row with model index {}'.format(idx))
        logger.info('\t'+ row.to_string().replace('\n', '\n\t')) 

        percentage = round(row['probability'].values[0],3)
        task_cov = row['tasks'].values[0]#[1:-1]

        logger.info('\npercentage: {}'.format(percentage))

        try:
            task_cov = task_cov.replace('\n','').split(' ')
        except:
            task_cov = []
        # task_cov = [int(cov) for cov in task_cov if cov != '']
        logger.info('tasks: {}'.format(task_cov))

        ##################
        # extract runtime
        ##################
        latency = row['latency'].values[0] #float(content[0].split(':')[1].replace(' ','').replace('\n',''))
        # inference = float(content[4].split(':')[1].replace(' ','').replace('\n',''))
        # step = int(row['step'])
        runtime = int(base + latency) #int(base) # #base * 16 #(int(inference) - base) * 15 + int(inference)

        logger.info('base cost: {}, latency: {}, total cost: {}'.format(base,latency,runtime))

        ##################
        # get selectivity for models
        ##################
        sel_model_repository = get_metrics(sel_model_repository,model,num_run_dir,stem,selectivity_dir,task_cov,'selectivity',selectivity=True)

        accu_model_repository = get_metrics(accu_model_repository,model,num_run_dir,stem,metric_dir,task_cov,metric,selectivity=False)
        accu_model_repository[model]['cost'] = runtime

    if all: 
        file_addition += '_all'
    # write_files(sel_model_repository,'../repository/'+dataset+'_selectivity_'+file_addition+'.csv')
    write_files(accu_model_repository,'../repository/'+dataset+'_model_stats_'+metric+'_new_model_'+file_addition+'.csv')

    logger.info('output path: {}'.format('../repository/'+dataset+'_model_stats_'+metric+'_new_model_'+file_addition+'.csv'))
    

def getPareto(dataset,metric,file_addition,synthetic=False):
    path = '../repository/'+dataset+'_model_stats_'+metric+'_new_model_'+file_addition+'.csv'
    pareto_path = '../repository/'+dataset+'_model_stats_'+metric+'_pareto_'+file_addition+'.csv'
    df = pd.read_csv(path,index_col=0)
    df_mask = util.getParetoSummary(df,synthetic=False,path=pareto_path)
    _ = util.getParetoModelOnly(df,df_mask,synthetic=False,path=pareto_path)

    logger.info('\n======== Pareto distribution =============')
    logger.info('origin input file of model repository: {}'.format(path))
    logger.info('output file of pareto model repository: {}'.format(pareto_path))

if __name__ == "__main__":

    logger = get_logger(__name__, '../logs/model_repository.log', use_formatter=False)

    # Parameters
    dataset = 'voc'  # coco voc
    for level in ['high']: #,'low','medium'
        for coverage in [30]: #[10,15,20,25,30]
            metric_dir = 'metrics_'+dataset+'_'+level # 'metrics_0.7'
            selectivity_dir = 'selectivity_' +dataset+'_'+ level
            file_addition = str(coverage)+'_'+level

            path = '../convert/'+dataset+'_model_config_new_model_'+file_addition+'.csv'

            # metric = 'accuracy' #'f1','ap','recall'
            for metric in ['accuracy','f1','precision','recall']:

                
                # get validation dataset len
                # num_imgs = len(os.listdir('/home/zli/experiments/datasets/coco128/images/train2017'))
                # num_imgs = len(os.listdir('/home/zli/experiments/datasets/coco/val'))
                
                getMR(path,dataset,coverage,metric,metric_dir,selectivity_dir,file_addition,all=False)
                getPareto(dataset,metric,file_addition)