import numpy as np
import pandas as pd
import random


def generateRandomRepository(mdist):
	# Selectivity
	M = 10
	N = 10
	s = np.random.uniform(0.3,0.9,N)
	# Mapping
	task_model_dict = {}
	model_split = split(M,size)
	task_split = split(N,size)

	for i in range(size):
		task_model_dict = mapping(task_split[i], model_split[i], task_model_dict)

	# generate df from dict
	df = pd.DataFrame.from_dict(task_model_dict, orient='index')
	num_assignment_each_model = list(df.count())
	length = len(num_assignment_each_model)
	max_count = max(num_assignment_each_model)
	if method =='max':
		_acc = df.max()
	else: _acc = df.median()
	df = df.T
	# print(df.head())
	# print('length of model repository:',len(df))
	# print('number of task:',len(df.columns))

	# Cost (5-100)
	low = 5
	high = 100
	medium = (low+high)//2
	
	# Select random 20% of the model to have random cost ranging from 60 to 100
	# sample_idx= random.sample(range(length),length//5)
	cost = [np.random.uniform(low,high) if _acc[i]<0.95 else np.random.uniform(medium,high) for i in range(length) ]
	# cost = np.random.uniform(low,high,length)
	df['cost'] = cost
	df.to_csv('repository/model_repository_'+str(M)+'_'+str(N)+'_'+method+'.csv')
	
	return df

def mapping(task_sub, model_sub, task_model_dict,low=0.7,high=0.98):
	for task in task_sub:
		task_model_dict['T'+str(task)] = {}
		sample = random.sample(model_sub,random.randint(1,len(model_sub)))
		selected_model_dict = {}
		for m in sample:
			if np.random.uniform()<0.1:
				selected_model_dict['M'+str(m)] = np.random.uniform(0.95,high)
			else: 
				selected_model_dict['M'+str(m)] = np.random.uniform(0.7,0.95)
		task_model_dict['T'+str(task)] = selected_model_dict
	return task_model_dict


def split(a, n):
	a_range = range(a)
	k, m = divmod(a, n)
	return [a_range[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]




if __name__ == '__main__':
	M = 100
	N = 40
	import timeit

	result = []
	M_list = [100 * 4 ** i for i in range(1,6)]
	N_list = [40,80,100,200]
	for M in M_list:
		for N in N_list:
			start = timeit.default_timer()
			generateRandomRepository(M,N,size=8)
			end = timeit.default_timer()
			print(M,N,end-start)
			result.append([M,N,end-start])

	df = pd.DataFrame(result,columns=['M','N','time'])
	df.to_csv('../simulation/synthesize_summary.csv')