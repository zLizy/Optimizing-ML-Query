import numpy as np
import random
import math
import pandas as pd



class Expression(object):
    OPS = ['&', '|']

    GROUP_PROB = 0.3
    NEGATION_PROB = 0.1
    a = 5

    nms_freq = ['person','chair','car','dinning_table','cup','bottle','bowl']
    nms = ['toaster', 'hair_drier', 'scissors', 'toothbrush', 'parking_meter', 'bear', 'snowboard', \
        'hot_dog', 'microwave', 'donut', 'sheep', 'stop_sign', 'broccoli', 'apple', 'carrot', 'frisbee', \
        'orange', 'zebra', 'fire_hydrant', 'cow', 'mouse', 'elephant', 'kite', 'teddy_bear', 'airplane', \
        'baseball_bat', 'sandwich', 'baseball_glove', 'giraffe', 'refrigerator', 'banana', 'suitcase', \
        'keyboard', 'wine_glass', 'oven', 'skis', 'boat', 'cake', 'bird', 'skateboard', 'horse', 'vase', \
        'remote', 'tie', 'bicycle', 'toilet', 'bed', 'surfboard', 'spoon', 'pizza', 'fork', 'train', \
        'motorcycle', 'tennis_racket', 'sports_ball', 'potted_plant', 'umbrella', 'dog', 'knife', 'laptop', \
        'cat', 'sink', 'bus', 'traffic_light', 'couch', 'clock', 'tv', 'cell_phone', 'backpack', 'book', \
        'bench', 'truck', 'handbag', 'bowl', 'bottle', 'cup', 'dining_table', 'car', 'chair', 'person']

    MIN_NUM, MAX_NUM = 0, len(nms_freq)-1


    
    def __init__(self, N, num_pred, conj_pro, selected=[],qdist='uniform', _maxdepth=None, _depth=0):
        """
        maxNumbers has to be a power of 2
        """
        self.MAX_NUM = N-1
        self.mu, self.sigma = self.MAX_NUM//2, 1


        if _maxdepth is None:
            _maxdepth = math.log(num_pred, 2) - 1

        # Left
        if _depth < _maxdepth: #and random.randint(0, _maxdepth) > _depth:
            self.left = Expression(N,num_pred,conj_pro,qdist=qdist, _maxdepth=_maxdepth, _depth=_depth + 1,selected=selected)
        else:
            if qdist == 'power_law':
                randomInts = int(np.random.power(Expression.a)*self.MAX_NUM)
                while randomInts in selected:
                    randomInts = int(np.random.power(Expression.a)*self.MAX_NUM)
                self.left = self.nms[randomInts]
            elif qdist == 'gaussian':
                randomNums = np.random.normal(loc=mu, scale=sigma)
                randomInts = np.round(randomNums)
                self.left = self.nms[randomInts]
            else:
                randomInts = np.random.randint(low=Expression.MIN_NUM, high=self.MAX_NUM)
                while randomInts in selected:
                    randomInts = int(np.random.randint(low=Expression.MIN_NUM, high=self.MAX_NUM))
                self.left = self.nms[randomInts]
            selected.append(randomInts)
            # if np.random.rand() < Expression.NEGATION_PROB:
            #     self.left = '!{0}'.format(self.left)

        # Right
        if _depth < _maxdepth: # and random.randint(0, _maxdepth) > _depth:
            self.right = Expression(N,num_pred,conj_pro,qdist=qdist, _maxdepth=_maxdepth, _depth=_depth + 1,selected=selected)
        else:
            if qdist == 'power_law':
                randomInts = max(int(np.random.power(Expression.a)*self.MAX_NUM),self.MAX_NUM)
                while randomInts in selected:
                    randomInts = int(np.random.power(Expression.a)*self.MAX_NUM)
                print(selected)
                self.right = self.nms[randomInts]
            elif qdist == 'gaussian':
                randomNums = np.random.normal(loc=mu, scale=sigma)
                randomInts = np.round(randomNums)
                self.right = self.nms[randomInts]
            else:
                randomInts = np.random.randint(low=Expression.MIN_NUM, high=self.MAX_NUM)
                while randomInts in selected:
                    randomInts = int(np.random.randint(low=Expression.MIN_NUM, high=self.MAX_NUM))
                print(selected)
                self.right = self.nms[randomInts]
            selected.append(randomInts)
            # if np.random.rand() < Expression.NEGATION_PROB:
            #     self.right = '!{0}'.format(self.right)

        self.grouped = np.random.rand() < Expression.GROUP_PROB
        
        # assign operator & / |
        # self.operator = random.choice(Expression.OPS)
        if np.random.uniform() <= conj_pro:
            self.operator = '&'
        else: self.operator = '|'

    def __str__(self):
        s = '{0!s} {1} {2!s}'.format(self.left, self.operator, self.right)
        if self.grouped:
            return '({0})'.format(s)
        else:
            return s

class CNFnDNF(object):

    # num_pred = pow(2,i) 
    # query = str(CNFnDNF(N,num_pred,distr)) 
    # N: number of tasks
    # num_pred: number of predicates
    nms_freq = ['person','chair','car','cup','bottle','bowl','tv','bird']

    nms = ['toaster', 'hair_drier', 'scissors', 'toothbrush', 'parking_meter', 'bear', 'snowboard', \
        'hot_dog', 'microwave', 'donut', 'sheep', 'stop_sign', 'broccoli', 'apple', 'carrot', 'frisbee', \
        'orange', 'zebra', 'fire_hydrant', 'cow', 'mouse', 'elephant', 'kite', 'teddy_bear', 'airplane', \
        'baseball_bat', 'sandwich', 'baseball_glove', 'giraffe', 'refrigerator', 'banana', 'suitcase', \
        'keyboard', 'wine_glass', 'oven', 'skis', 'boat', 'cake', 'bird', 'skateboard', 'horse', 'vase', \
        'remote', 'tie', 'bicycle', 'toilet', 'bed', 'surfboard', 'spoon', 'pizza', 'fork', 'train', \
        'motorcycle', 'tennis_racket', 'sports_ball', 'potted_plant', 'umbrella', 'dog', 'knife', 'laptop', \
        'cat', 'sink', 'bus', 'traffic_light', 'couch', 'clock', 'tv', 'cell_phone', 'backpack', 'book', \
        'bench', 'truck', 'handbag', 'bowl', 'bottle', 'cup', 'dining_table', 'car', 'chair', 'person']

    def __init__(self, num_pred, distr='uniform', form='CNF',_depth=0):
        """
        maxNumbers has to be a power of 2
        """
        self.MAX_NUM = len(self.nms_freq)-1
        if synthetic:
            self.MAX_NUM = 1930
        self.MIN_NUM = 0
        self.num_pred = num_pred
        self.distr = distr
        self.a = 5

        if form == 'CNF':
            self.connect_opr = ' & '
            self.inter_opr = ' | '
        else:
            self.connect_opr = ' | '
            self.inter_opr = ' & '

        # self.get(num_pred)

    def __str__(self):
        return self.get(self.num_pred)
        
    def get(self,num_pred):
        components = []
        selected = []

        while num_pred > 0:
            print('num_pred',num_pred)
            subquery,size,selected = self.generate(num_pred,self.inter_opr,selected)
            num_pred -= size
            if size == 0:
                continue
            components.append((subquery,size))

        s = self.combine(components,self.connect_opr)
        return s

    def genComponents(self,num_pred,operator,selected,low):
        size = 2#np.random.randint(low=low,high=num_pred)
        print('size',size)
        if self.distr == 'power_law':
            pred = []
            # while size > 0:
            for i in np.random.power(self.a,size):
                if int(i*self.MAX_NUM) not in selected:
                    tmp = min(int(i*self.MAX_NUM),self.MAX_NUM)
                    pred += [tmp]
                    selected += [tmp]
        else:
            pred = []
            while size > 0:
                tmp = np.random.randint(low=self.MIN_NUM, high=self.MAX_NUM)
                if tmp not in selected:
                    size -= 1
                    selected += [tmp]
                    pred += [tmp]
            # pred = [i for i in pred if i not in selected]
        # selected += pred
        print('pred',pred)
        if synthetic:
            predicates = ['P'+str(i) for i in pred]
        else:
            predicates = [self.nms_freq[i] for i in pred]
        print('predicates',predicates)
        subquery = operator.join(predicates)
        print('subquery',subquery)
        return subquery,len(pred),selected

    def generate(self,num_pred,operator,selected):
        if num_pred == 1:
            size = 1
            if self.distr == 'power_law':
                pred = min(int(np.random.power(self.a)*self.MAX_NUM),self.MAX_NUM)
                while pred in selected:
                    pred = min(int(np.random.power(self.a)*self.MAX_NUM),self.MAX_NUM)
            else:
                pred = np.random.randint(low=self.MIN_NUM,high=self.MAX_NUM)
                while pred in selected:
                    pred = np.random.randint(low=self.MIN_NUM,high=self.MAX_NUM)
                selected += [pred]
            if synthetic:
                return 'P'+str(pred), size, selected
            else:
                return self.nms_freq[pred],size,selected
        # elif num_pred == 2:
        #     return self.genComponents(num_pred,operator,selected,1)
        else:
            return self.genComponents(num_pred,operator,selected,2)

    def combine(self,components,operator):
        # components: [(subquery,size)]
        s = ['('+part[0]+')' if part[1] != 1  else part[0] for i,part in enumerate(components) ]
        s = operator.join(s)
        print('s',s)
        return s

if __name__ == '__main__':

    M = 1280
    N = 1922
    num_ops = 20
    num_exp = 5
    num_iter = 2
    conj_pro = 0.5
    distr = 'uniform'
    form = ['cnf','dnf']
    selected = []

    synthetic = 1

    df = pd.DataFrame(columns=['query','#predicate','form','#&','#|','#()','ratio_&','ratio_|'])
    for f in form:
        for i in range(1,num_exp+1):
            # print(Expression(N,pow(2,random.randint(1,5))).__str__())
            for j in range(num_iter):
                num_pred = pow(2,i)
                selected = []
                # query = str(Expression(N,num_pred,conj_pro,qdist=distr,selected=selected))
                query = str(CNFnDNF(num_pred,form=f,distr=distr))
                print(query)
                print(selected)
                num_and = len(query.split('&'))-1
                num_or = len(query.split('|'))-1
                num_par = len(query.split('('))-1
                # print('number of &: ',num_and)
                # print('number of |: ',num_or)
                # print('number of (: ',num_par)
                item = {'query':query,'#predicate':num_pred,'form':f,\
                    '#&':num_and,'#|':num_or,'#()':num_par, \
                    'ratio_&':round(num_and/(num_pred-1),4), \
                    'ratio_|':round(num_or/(num_pred-1),4)}
                df = df.append(item,ignore_index=True)
        print(df)
        path = 'coco_query_'+distr+'_freq.csv'
        if synthetic:
            path = 'synthetic_query_'+distr+'.csv'
    df.to_csv(path)#,index=False)

