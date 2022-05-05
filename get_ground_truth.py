from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pandas as pd
import numpy as np
import skimage.io as io
import os
import pylab
from sympy.parsing.sympy_parser import parse_expr
pylab.rcParams['figure.figsize'] = (10.0, 8.0)
# os.system('clear')

def get_coco_tool():
    annType = ['segm','bbox','keypoints']
    annType = annType[1]      #specify type here
    prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
    print('Running demo for *%s* results.'%(annType))

    #initialize COCO ground truth api
    dataDir='/home/zli/experiments/datasets/coco/data'
    dataType='val2017'
    annFile = '%s/annotations/%s_%s.json'%(dataDir,prefix,dataType)
    coco=COCO(annFile)
    return coco

def get_file(filepath):
    if os.path.isfile(filepath):
        # print ("Query file exist")
        df_query = pd.read_csv(filepath,index_col=0)
    #     df_query = df_query[df_query['form']=='dnf']
        # queryList = list(df_query['query'])
        # indexList = list(df_query['index'])#list(df_query['index'])#[json.loads(l) for l in list(df_query['index'])]
        return df_query

def compare(gt_file,result_dir,sub_file,q_type,query_idx,bound,repository_file):

    filepath='simulation/coco_query_'+q_type+'_gt.csv'
    df_query = get_file(filepath)
    df_query = df_query.iloc[query_idx]
    query = df_query['query']

    predicates = [str(p) for p in list(parse_expr(query).atoms())]

    df_gt = get_file(gt_file)

    df_result = get_file(result_dir+sub_file)

    # get scripts
    script_file = 'script_new_inference/'+sub_file.replace('.csv','.sh')
    print('###### query idx: '+ str(query_idx)+ ', query: ' + query + ', bound: '+ str(bound) +' ###########')
    print()

    with open(script_file) as fp:
        line = fp.readline()
        cn = 1
        while line:
            script = line.split('\n')[0]+ ' --verify --q_type ' + q_type + ' --query_idx ' + str(query_idx) + ' --bound '+ str(bound) + ' --repository_file ' + repository_file
            print("Line {}: {}".format(cn,script))
            os.system(script)
            cn+=1
            line = fp.readline()


if __name__ == '__main__':

    GT_dir = '/home/zli/experiments/datasets/coco/data/annotations/'
    GT_file = GT_dir + 'val2017.csv'

    metric = 'ap' # 'f1'
    q_type = 'dnf'
    query_idx = 3
    num_dict = {2:'4',3:'5',4:'6'}
    num_predicate = num_dict[query_idx]
    cost = 75

    coverage = '30'
    level = 'medium'

    for cost in [75,100]: #[75,100,125,150,175,200]:
        for level in ['medium']:
            result_dir = '/home/zli/experiments/Project_Boolean_simple/output/tables/'
            config_dir = 'coco_'+metric+'_'+q_type+'_cost_'+coverage+'_'+ level +'/'
            result_file = 'script_order_'+str(query_idx)+'_'+q_type+'_'+num_predicate+'_cost_'+str(cost)+'.csv'
            repository_file = 'repository/model_stats_'+metric+'_new_model_'+coverage+'_'+level+'.csv'
            print('=========== '+ level +' =============')
            try:
                compare(GT_file,result_dir,config_dir+result_file,q_type,query_idx,cost,repository_file)
            except:
                continue