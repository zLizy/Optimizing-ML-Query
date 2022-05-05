import os
import pandas as pd
from sympy import *
from sympy.parsing.sympy_parser import parse_expr


def getQueryNum(df,df_label):
    index_dic = {}
    count_dic = {}
    form_dic = {}
    query_list = df['query']
    print(len(query_list))

    for ind, query in enumerate(query_list):
        print('########\n')
        if ind % 5 == 0:
            print(ind)
        print(query)
        Bxp = parse_expr(query)
        predicates = [str(p) for p in list(Bxp.atoms())]
        print(predicates)

        subset = df_label[predicates].dropna(how='all')
        subset = pd.notnull(subset)

        index = []
        for i, row in subset.iterrows():
            if Bxp.subs({p:row[str(p)] for p in list(Bxp.atoms())}) == True: 
                try:
                    index.append(df_label.loc[i,:]['filename'])
                except:
                    continue
        # index = [df_label.iloc[i,:]['filename'] for i, row in subset.iterrows() if Bxp.subs({p:row[str(p)] for p in list(Bxp.atoms())}) == True ]
        print('len(index)',len(index))
        
        index_dic[query] = index
        count_dic[query] = len(predicates)
        form_dic[query] = df[df['query']==query]['form'].values[0]
    return index_dic,count_dic,form_dic

def saveFile(index_dic,count_dic,form,path):
    df_index = pd.DataFrame(columns=['query','#predicate','len','index'])
    df_index['query'] = index_dic.keys()
    df_index['#predicate'] = count_dic.values()
    df_index['len'] = [len(v) for v in index_dic.values()]
    df_index['index'] = index_dic.values()
    df_index['form'] = form.values()
    df_index.to_csv(path)#query_uniform_gt.csv')
    print(df_index.head())
    return df_index


data = 'coco'
data_name = 'coco_voc'

# query file
if data_name == 'voc' or data_name == 'coco_voc':
    # query
    path = 'voc_query_uniform_freq.csv'
elif data_name == 'coco':
    # path = 'voc_query_uniform_freq.csv'
    path = 'coco_query_uniform_freq.csv'

df = pd.read_csv(path)

# ground truth
if data == 'coco':
    # coco
    dataDir = '../../datasets/coco'
    dataType='val2017'
    annFile='{}/data/annotations/instances_{}.json'.format(dataDir,dataType)
    resultFile = '{}/data/annotations/{}.csv'.format(dataDir,dataType)
    # categoryFile = '{}/annotations/label_val.csv'.format(dataDir)
    categoryFile = '{}/data/annotations/category_mapping.csv'.format(dataDir)
    test_img_files = os.listdir('/home/zli/experiments/datasets/coco/images/test')
elif data == 'voc':
    resultFile = '/home/zli/experiments/datasets/voc_2012/gt.csv'
    test_img_files = os.listdir('/home/zli/fiftyone/voc_2012_test/data')

df_cnf = df[df['form']=='CNF']
df_dnf = df[df['form']=='DNF']

df_label = pd.read_csv(resultFile,index_col=0)

# select test images

df_label = df_label[df_label['filename'].isin(test_img_files)]

d_cnf,count_dic,form_cnf = getQueryNum(df_cnf,df_label)
df_cnf = saveFile(d_cnf,count_dic,form_cnf,data_name+'_query_cnf_gt.csv')

d_dnf,count_dic,form_dnf = getQueryNum(df_dnf,df_label)
df_dnf = saveFile(d_dnf,count_dic,form_dnf,data_name+'_query_dnf_gt.csv')