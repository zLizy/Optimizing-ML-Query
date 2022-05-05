import numpy as np
import random
import math
import pandas as pd



class Expression(object):
    OPS = ['&', '|']

    GROUP_PROB = 0.3
    NEGATION_PROB = 0.1
    a = 5

    MIN_NUM, MAX_NUM = 0, 40
    
    def __init__(self, N, num_pred, conj_pro, qdist='uniform', _maxdepth=None, _depth=0):
        """
        maxNumbers has to be a power of 2
        """
        self.MAX_NUM = N-1
        self.mu, self.sigma = self.MAX_NUM//2, 1

        if _maxdepth is None:
            _maxdepth = math.log(num_pred, 2) - 1

        # Left
        if _depth < _maxdepth: #and random.randint(0, _maxdepth) > _depth:
            self.left = Expression(N,num_pred,conj_pro,qdist, _maxdepth, _depth + 1)
        else:
            if qdist == 'power_law':
                randomInts = int(np.random.power(Expression.a)*self.MAX_NUM)
                self.left = 'T'+str(randomInts)
            elif qdist == 'gaussian':
                randomNums = np.random.normal(loc=mu, scale=sigma)
                randomInts = np.round(randomNums)
                self.left = 'T'+str(randomInts)
            else:
                self.left = 'T'+str(np.random.randint(low=Expression.MIN_NUM, high=self.MAX_NUM))
            # if np.random.rand() < Expression.NEGATION_PROB:
            #     self.left = '!{0}'.format(self.left)

        # Right
        if _depth < _maxdepth: # and random.randint(0, _maxdepth) > _depth:
            self.right = Expression(N,num_pred,conj_pro, qdist,_maxdepth, _depth + 1)
        else:
            if qdist == 'power_law':
                randomInts = int(np.random.power(Expression.a)*self.MAX_NUM)
                self.right = 'T'+str(randomInts)
            elif qdist == 'gaussian':
                randomNums = np.random.normal(loc=mu, scale=sigma)
                randomInts = np.round(randomNums)
                self.right = 'T'+str(randomInts)
            else:
                self.right = 'T'+str(np.random.randint(low=Expression.MIN_NUM, high=self.MAX_NUM))
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




if __name__ == '__main__':

    M = 100
    N = 40
    num_ops = 20
    num_exp = 5
    num_iter = 10
    conj_pro = 0.6
    distr = 'gaussian'

    df = pd.DataFrame(columns=['query','#predicate','#&','#|','#()','ratio_&','ratio_|'])
    for i in range(1,num_exp+1):
        # print(Expression(N,pow(2,random.randint(1,5))).__str__())
        for j in range(num_iter):
            num_pred = pow(2,i)
            query = str(Expression(N,num_pred,conj_pro,distr))
            print(query)
            num_and = len(query.split('&'))-1
            num_or = len(query.split('|'))-1
            num_par = len(query.split('('))-1
            print('number of &: ',num_and)
            print('number of |: ',num_or)
            print('number of (: ',num_par)
            item = {'query':query,'#predicate':pow(2,i),\
                '#&':num_and,'#|':num_or,'#()':num_par, \
                'ratio_&':round(num_and/(num_pred-1),4), \
                'ratio_|':round(num_or/(num_pred-1),4)}
            df = df.append(item,ignore_index=True)
    print(df)
    df.to_csv('query.csv',index=False)

