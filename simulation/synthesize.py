import numpy as np
import pandas as pd
import random


def generateRandomRepository(M,N,size):

	# Cost
	c1 = np.random.uniform(5,20,M//3)
	c2 = np.sort(np.random.uniform(20,40,M//3))
	c3 = np.random.uniform(40,60,M-(M//3)-(M//3))

	# Selectivity
	s = np.random.uniform(0.3,0.9,N)

	# models divided into 2 groups
	g1 = c1.tolist() + c2.tolist()[:len(c1)//2]
	g2 = c2.tolist()[len(c1)//2:] + c3.tolist()
	print(len(g1),len(g2))

	# modelCost
	modelCost = g1+g2

	# Mapping
	l = [g1,g2]
	model_count = 0
	task_count = 0
	task_model_dict = {}
	for i in range(2):
	    # size = 5
	    if i==0: 
	        task_idx = 0
	        task_end = N//2
	    else: 
	        task_idx = N//2
	        task_end = N
	    task_range = range(task_idx, task_end)
	    print(task_range)
	    task_range_sub = list(split(task_range, size))
	    print(task_range_sub)
	    model_range_sub = list(split(l[i],size))
	    # print(model_range_sub)
	    for j in range(size):
	        print('model count:'+str(model_count))
	        print('task count:'+str(task_count))
	        mapping(task_model_dict,model_range_sub[j],task_range_sub[j],model_count,task_count,size)
	        model_count += len(model_range_sub[j])
	        task_count += len(task_range_sub[j])
	        

	# print(task_model_dict)

	# generate df from dict
	df = pd.DataFrame.from_dict(task_model_dict, orient='index')
	df = df.T
	print(len(df),len(modelCost))
	print(df.head())
	cost = list(np.random.uniform(5,60,len(df)))
	df['cost'] = cost
	df.to_csv('repository/model_repository_'+str(M)+'_'+str(N)+'.csv')
	
	
	return df


def mapping(task_model_dict,cluster,task_range,model_lower=0,task_lower=0,size=3):
    # average cost
    max_cost = max(cluster)
    for idx,item in enumerate(task_range):
        task_model_dict['T'+str(idx+task_lower)]={}
        ## assignment #count of models to a task [1,min(size,#models in cluster)]
        count = min(random.randint(1,size),len(cluster))
        selected_idx = random.sample(range(len(cluster)), count)
        # random_model = np.random.randint(cluster[0],cluster[-1]+1,count)
        # random_task = np.random.randint(task_range[0],task_range[-1]+1,count)
        selected_model_dict = {}
        print(selected_idx)
        for i in selected_idx:
        	# print('model idx:',str(model_lower+i))
        	# costList.append(cluster[i])
        	selected_model_dict['M'+str(model_lower+i)] = np.random.uniform(0.93*cluster[i]/max_cost,0.98*cluster[i]/max_cost)
        task_model_dict['T'+str(idx+task_lower)] = selected_model_dict
    return task_model_dict

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))




if __name__ == '__main__':
	M = 100
	N = 40
	task_model_dict = {}
	generateRandomRepository(M,N)