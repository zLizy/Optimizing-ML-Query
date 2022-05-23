# most borrow from: https://github.com/HCPLab-SYSU/SSGRL/blob/master/utils/metrics.py
import os 
import re
import numpy as np
import sys
import pandas as pd
sys.path.append('../test')
import tensorflow as tf
import glob
from keras import backend as K
# import logging
from my_logger import get_logger


# df_logger = get_logger(str(__name__)+'_dfs', 'logs/debug.log', use_formatter=False)


def voc_ap(rec, prec, true_num):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_mAP(output,gt_label, num=80, return_each=False):
    
    num_target = np.sum(gt_label, axis=1, keepdims = True)

    sample_num = len(gt_label)
    # print(sample_num)
    class_num = num
    tp = np.zeros(sample_num)
    fp = np.zeros(sample_num)
    aps = []

    for class_id in range(class_num):
        confidence = output[:,class_id]
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        sorted_label = [gt_label[x][class_id] for x in sorted_ind]

        for i in range(sample_num):
            tp[i] = (sorted_label[i]>0)
            fp[i] = (sorted_label[i]<=0)
        true_num = 0
        true_num = sum(tp)
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(true_num)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = np.round(voc_ap(rec, prec, true_num),2)
        aps += [ap]

    np.set_printoptions(precision=6, suppress=True)
    aps = np.round(np.array(aps),3) #* 100
    mAP = np.mean(aps)

    if return_each:
        return mAP, aps
    return aps

def AUC(predict,target):
    m = tf.keras.metrics.AUC(num_thresholds=11)
    m.update_state(target,predict)
    return np.round(m.result().numpy(),3)

def binary_accuracy(predict,target, class_num=80):
    m = tf.keras.metrics.BinaryAccuracy()
    m.update_state(target,predict)
    return np.round(m.result().numpy(),3)

def precision(predict,target):
    m = tf.keras.metrics.Precision()
    m.update_state(target,(predict>0).astype(int))
    return np.round(m.result().numpy(),3)

def recall(predict,target):
    m = tf.keras.metrics.Recall()
    m.update_state(target,(predict>0).astype(int))
    return np.round(m.result().numpy(),3)


def keras_metrics(output,gt_label,class_num=80):

    Accu = []
    # bi_accu = []
    AUCs = []
    precisions = []
    recalls = []
    f1s = []
    for class_id in range(class_num):
        predict = output[:,class_id]
        target = gt_label[:,class_id]

        accu = binary_accuracy(predict,target)
        Accu.append(accu)

        predict = np.reshape(predict,len(predict))
        target = np.reshape(target,len(target))
        
        auc = AUC(predict,target)
        AUCs.append(auc)

        pre = precision(predict,target)
        precisions.append(pre)

        re = recall(predict,target)
        recalls.append(re)
        # print(pre,re)

        f1 = 2*((pre*re)/(pre+re+K.epsilon()))
        f1s.append(f1)
        # print('accu','auc','pre','recall','f1')
        # print(accu,auc,pre,re,f1)

    return Accu, AUCs, precisions, recalls, f1s

def create_dir(out_dir,addition):
    if not isinstance(out_dir,list):
        if not os.path.exists(out_dir+addition):
            os.makedirs(out_dir+addition)
    else:
        for o_dir in out_dir:
            create_dir(o_dir,addition)

def write_selectivity(df_variants,sel_file):
    df_variants = df_variants.drop(columns=['filename'])
    s_df = round(df_variants.sum(axis=0)/len(df_variants),4)
    s_df = s_df.to_frame(name='selectivity')
    s_df['class'] = s_df.index
    s_df.index = range(len(s_df))
    s_df.to_csv(sel_file)

def get_path(out_dir,count,index,base_cost):
    return out_dir+'/'+str(count)+'/summary_'+str(index)+'_'+str(base_cost)+'.csv'

def get_variants(df,df_config,model_name,base_cost,out_dir,gt_label,img_files,classes=[],test=False,run=1):
    
    for count in range(run):
        create_dir(out_dir,'/'+str(count))

    if test:
        fill_val = np.NaN
    else:
        fill_val = np.NaN
        out_dir,metric_dir,selectivity_dir = out_dir
        
    # print(df_config)
    for i,row in df_config.iterrows():
        logger.info('------------')
        logger.info('df_config row of model {}'.format(model_name))
        # logger.debug(row)
        logger.info('\t'+ row.to_string().replace('\n', '\n\t')) 

        index = row['index']
        model_name = row['model']
        probability = row['probability']
        for count in range(run):
            df_variants = df.copy()
            for cl in classes:
                # sampling p proportion of the original output and set the rest as 'fill_val' (np.NaN)
                df_variants.loc[df[cl].sample(frac=1-probability).index,cl] = fill_val
            
            # df_variants['filename'] = df['filename']
            if not test:
                metric_file = get_path(metric_dir,count,index,base_cost)
                sel_file = get_path(selectivity_dir,count,index,base_cost)
            out_file = get_path(out_dir,count,index,base_cost)

            if test:
                df_variants.to_csv(out_file)
            else:
                df_variants.to_csv(out_file)
                write_metrics(df_variants,metric_file,gt_label,img_files,classes)
                write_selectivity(df_variants,sel_file)

def write_metrics(df,file,gt_label,img_files,nms):
    

    # logger.info('\n--- Calculating metrics ---')

    df = df[nms].fillna(0)
    # print(df.shape)
    aps = voc_mAP(df.to_numpy(),gt_label, num=len(nms), return_each=False)
    accu, AUCs, precisions, recalls, f1s = keras_metrics(df.to_numpy(),gt_label,class_num=len(nms)) # Accu, AUCs, precisions, recalls, f1s

    # print(len(aps),len(accu),len(AUCs),len(precisions),len(recalls),len(f1s))

    # with open('output/'+file,'r') as f:
    #     lines = f.readlines()

    # logger.info('class: {}, AP: {}, accuracy: {}, AUC: {}, precision: {}, recall: {}, f1: {}'.format(nms,aps,accu,AUCs,precisions,recalls,f1s))

    df_result = pd.DataFrame(columns=['class','AP','accuracy','AUC','precision','recall','f1'])
    df_result['class'] = nms
    df_result['AP'] = aps
    df_result['accuracy'] = accu
    df_result['AUC'] = AUCs
    df_result['precision'] = precisions
    df_result['recall'] = recalls
    df_result['f1'] = f1s
    df_result.to_csv(file)

def main(dataset,val_file,out_dir,base_file,config_file,gt_file,run=10):

    logger.info('====================')
    logger.debug('Test: {}'.format(test))
    logger.info('dataset: {}'.format(dataset))
    logger.info('val_file: {}'.format(val_file))
    logger.info('out_dir: {}'.format(out_dir))
    logger.info('base_file: {}'.format(base_file))
    logger.info('config_file: {}'.format(config_file))
    logger.info('gt_file: {}'.format(gt_file))

    class_num = 80

    if dataset == 'coco':
        nms = ['person','bicycle','car','motorcycle','airplane','bus','train',\
                    'truck','boat','traffic_light','fire_hydrant','stop_sign','parking_meter',\
                    'bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra',\
                    'giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis',\
                    'snowboard','sports_ball','kite','baseball_bat','baseball_glove','skateboard',\
                    'surfboard','tennis_racket','bottle','wine_glass','cup','fork','knife','spoon',\
                    'bowl','banana','apple','sandwich','orange','broccoli','carrot','hot_dog','pizza',\
                    'donut','cake','chair','couch','potted_plant','bed','dining_table','toilet','tv',\
                    'laptop','mouse','remote','keyboard','cell_phone','microwave','oven','toaster',\
                    'sink','refrigerator','book','clock','vase','scissors','teddy_bear','hair_drier','toothbrush']
    elif dataset == 'voc':
        nms = ['airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                'car', 'cat', 'chair', 'cow', 'dining_table', 'dog', 'horse', 'motorcycle',
                'person', 'potted_plant', 'sheep', 'couch', 'train', 'tv']
    
    # img_files = pd.read_csv('tables/summary_26_50.csv',index_col=0)['filename']
    img_files = pd.read_csv(glob.glob(val_file+'/summary_1_yolov3_*.csv')[0],index_col=0)['filename']

    model_files = os.listdir(val_file+'/')
    df_config = pd.read_csv(config_file,index_col=0)
    
    try:
        if not os.path.exists(out_dir+'/'):
            os.makedirs(out_dir)
    except:
        for o_dir in out_dir:
            if not os.path.exists(o_dir+'/'):
                os.makedirs(o_dir)
    
    df_label_all = pd.read_csv(gt_file,index_col=0)
    df_label_val = df_label_all[df_label_all['filename'].isin(img_files)]
    print()
    df_label_val = df_label_val.set_index('filename')
    df_label_val = df_label_val.loc[img_files].fillna(0)
    # print(df_label_val.head())
    gt_label = df_label_val[nms].to_numpy()
    # print(df_label_val.shape,'gt_label:',len(gt_label))

    for file in model_files:
        print(file)
        logger.info('~~~~~~~~~~~~~~~~~~~')
        logger.info('model file: {}'.format(val_file+'/'+file))
        
        df = pd.read_csv(val_file+'/'+file,index_col=0)
        df = df[df['filename'].isin(img_files)]
        # print(df.shape)

        parts = re.compile(r'summary_\d+_(\w*)_(\d+).csv').match(file)
        model_name = parts.group(1)
        base_cost = parts.group(2)
        logger.info('model_name {}'.format(model_name))
        logger.info('base_cost {}'.format(base_cost))

        get_variants(df,df_config[df_config['model']==model_name],model_name,base_cost,out_dir,gt_label,img_files,classes=nms,test=test,run=run)
        
        
        # with open('output/'+file, 'w') as f:
        #     f.write('class\tAP\taccuracy\tAUC\tprecision\trecall\tf1\n')
        #     for i in range(class_num):
        #         f.write('{label}\t{ap}\t{accu}\t{AUC}\t{precision}\t{recall}\t{f1}\n'
        #                 .format(label=nms[i],ap=aps[i],accu=accu[i],AUC=AUCs[i],precision=precisions[i],recall=recalls[i],f1=f1s[i]))
                # f.write('{prev} {metrics}\n'.format(prev=lines[i][:-1],metrics=metrics[i]))



if __name__ == '__main__':
    
    test = False
    if test:
        suffix='test'
    else:
        suffix='val'
        
    logger = get_logger(__name__, '../logs/voc_inference_'+suffix+'.log', use_formatter=False)
    
    all = False
    run = 10
    data = 'voc' # coco voc
    if data == 'coco':
        gt_file = '/home/zli/experiments/datasets/coco/data/annotations/val2017.csv'
    elif data == 'voc':
        gt_file = '/home/zli/experiments/datasets/voc_2012/gt.csv'

    for level in ['high']: #['high','low','medium']
        if data == 'voc':
            config_file = '../convert/voc_model_config_new_model_30_'+level+'.csv'
        else:
            config_file = '../convert/model_config_new_model_30_'+level+'.csv'
        # prob = re.compile(r'.*_(0.\d+).csv').match(config_file).group(1)

        if test:
            base_file = 'base_'+data+'_test.csv'
            val_file = 'raw_'+data+'_test'
            out_dir = 'inference_'+data+'_'+level
        elif all:
            # evaluate all
            base_file = 'base_all.csv'
            val_file = 'raw_all'
            metric_dir = 'metrics_all_'+level
            selectivity_dir = 'selectivity_all_'+level
            out_dir = ['val_all_'+level,metric_dir,selectivity_dir]
        elif not test:
            base_file = 'base_'+data+'_val.csv'
            val_file = 'raw_'+data+'_val'
            metric_dir = 'metrics_'+data+'_'+level
            selectivity_dir = 'selectivity_'+data+'_'+level
            out_dir = ['val_'+data+'_'+level,metric_dir,selectivity_dir]
        

        main(data,val_file,out_dir,base_file,config_file,gt_file,run=run)

    