import random
import time
import os.path
from util import *
import pandas as pd
from sympy.parsing.sympy_parser import parse_expr
import argparse
import matplotlib.pyplot as plt
from baselines.baseline_c_cost import getBaseline1
from baselines.baseline_c_accuracy_reverse import getBaseline2



def run(args):

	countModel = 0
	countPareto = 0
	tempList = []
	tempQuery = []

	if args.synthetic:
		# get model repository
		df = getRandomRepository(args.mdist,args.a)
		tasks = ['P'+str(idx) for idx in range(len(df.columns)-1)]
		df.columns = tasks+['cost']
		print('len of df is: '+str(len(df.columns)))

		models = ['M'+str(idx) for idx in range(len(df.index))]
		df.index = models

		# get Pareto summary
		df_pareto = getParetoSummary(df,args.mdist,args.a)
		# print(df_pareto.head())
		
		# synthesize queries
		df_query = getQueries(len(tasks),args.qdist)
		# queryList = getQueries(len(tasks),args.qdist)
	else:
		df = getYOLORepository()
		# df = df.drop(['model_44'])
		# df = df.drop(['model_30'])

		tasks = [c for c in df.columns if c != 'cost']
		models = df.index
		print('len of df is: '+str(len(df.columns)))
		# get Pareto summary
		df_pareto = getParetoSummary(df,synthetic=args.synthetic)
	
		# synthesize queries
		df_query = getQueries(synthetic=args.synthetic)
	
	# idx = 1
	_time = 0
	data_process_time = 0

	# get Pareto model only
	if args.approach == 'baseline':
		df = getParetoModelOnly(df,df_pareto,args.mdist,args.a,synthetic=args.synthetic)

	# queryList = queryList[:1] + queryList[2:]#[queryList[0]]#
	for idx,row in df_query.iterrows():
		if idx>3:
			print('query '+ str(idx))
		else:
			print('###################')
			query = row['query']
			try:
				query_size = str(row['#predicate'])
			except: query_size = '0'
			try:
				query_type = row['form']
			except:
				query_type = 'DNF'

			print('query '+ str(idx))
			print(query)
			T,steps = getSteps(str(query))

			# data process
			start = time.time()#timeit.default_timer()
			selected_col = [t.replace('_',' ') for t in T]
			try:
				df_selected = df[selected_col]
			except:
				print('model not exist to answer tasks')
				continue
			df_selected = df_selected.dropna(how='all').fillna(0)
			row_selected = df_selected.index
			M = df_selected.index.tolist()
			cost = df.loc[row_selected,'cost']
			Accuracy = df_selected.to_numpy()

			# selectivity = [0.8] * len(T)

			# selectivity
			if args.order:
				selectivity = getSelectivity(tasks,args.synthetic)
			end = time.time()#timeit.default_timer()
			data_process_time = end-start


			bound = args.bound
			# if args.constraint == 'cost':
				# bound = args.bound * len(T)
			print('bound',bound)

			# compute runtime
			# ## optimizer
			if args.approach == 'optimizer':
				start = time.time()#timeit.default_timer()
				if args.order:
					# from optimization.optimizer_order import Optimizer
					from optimization.optimizer_order_all import Optimizer
					optimizer = Optimizer(idx,str(query),steps,M,T,Accuracy,cost,selectivity,args.constraint,bound,query_type)

				else:
					from optimization.optimizer import Optimizer
					optimizer = Optimizer(steps,M,T,Accuracy,cost,args.constraint,bound)
				# assignment = {task:model}
				assignment,pre_order,_A,_C = optimizer.optimize()
				end = time.time()#timeit.default_timer()
				_time = end-start
				print(_A,_C)
				print('time',_time)
				print(assignment)
				
				if not args.synthetic:
					# print()
					Etype = 'opt'
					if args.record_test:
						Etype = 'test_'+Etype
					writeScript(assignment,str(query),idx,pre_order,Etype,args.constraint,args.bound,query_type,query_size,args.order)
				if args.record:
					writeIntermediateParetoSummary(args,idx,df_pareto,query,T,assignment,_A,_C,_time,data_process_time)
			else:
				Cost = np.array([[cost[i] if Accuracy[i,j] !=0 else 5000 for j in range(len(T))] for i in range(len(M))])
				if args.constraint == 'cost':
					## baseline1
					start = time.time() #timeit.default_timer()
					# task:model
					# shuffle order
					indices = np.arange(len(T))
					np.random.shuffle(indices)
					Cost = Cost[:,indices]
					T = list(np.array(T)[indices])
					Accuracy = Accuracy[:,indices]
					# run
					flag,_A,_C,assignment = getBaseline1(steps,M,T,Cost,Accuracy,bound,selected_model={})
					end = time.time() #timeit.default_timer()
					_time = end-start
					# print(flag,_A,_C)
					print(assignment)
					
					if args.order:
						pre_order = []
					else:
						pre_order = list(assignment.keys())

					if not args.synthetic:
						# print()
						Etype = 'base'
						if args.record_test:
							Etype = 'test_'+Etype
						writeScript(assignment,str(query),idx,pre_order,Etype,args.constraint,args.bound,query_type,query_size,args.order)
					if args.record:
						writeIntermediateParetoSummary(args,idx,df_pareto,query,T,assignment,_A,_C,_time,data_process_time,approach='baseline_pareto')
				else:
					## baseline2
					start = time.time()  # timeit.default_timer()
					from baselines.baseline_c_accuracy import getBaseline2

					flag,_A,_C,assignment = getBaseline2(steps,M,T,Cost,Accuracy,bound,start,selected_model={})
					if args.order:
						pre_order = []
					else:
						pre_order = list(assignment.keys())
					
					end = time.time() #timeit.default_timer()
					_time = end-start
					print(flag,_A,_C)
					print(assignment)
					
					if args.record:
						writeIntermediateParetoSummary(args,idx,df_pareto,query,T,assignment,_A,_C,_time,data_process_time,approach='baseline_pareto')

					if not args.synthetic:
						writeScript(assignment,str(query),idx,pre_order,'base',args.constraint,args.bound,query_type,query_size,args.order)

		# idx+=1	

	return _time


if __name__ == '__main__':
	
	# python run.py -mdist uniform -qdist uniform -constraint cost -bound 200
	# -synthetic -order -record
	# -approach baseline
	# python run.py -mdist uniform -qdist power_law -constraint cost -bound 200 -approach baseline
	# python run.py -mdist power_law -qdist uniform -constraint cost -bound 200 -approach baseline
	# -a 4
	# python run.py -mdist power_law -qdist power_law -constraint cost -bound 200 -approach baseline

	# python run.py -mdist uniform -qdist uniform -constraint accuracy -bound 0.95 -approach baseline
	# python run.py -mdist uniform -qdist power_law -constraint accuracy -bound 0.95 -approach baseline
	# python run.py -mdist power_law -qdist uniform -constraint accuracy -bound 0.95 -approach baseline
	# python run.py -mdist power_law -qdist power_law -constraint accuracy -bound 0.95 -approach baseline

	# COCO
	# python run.py -constraint accuracy -bound 0.85 -synthetic -record
	# python run.py -constraint cost -bound 50 -record -order -approach baseline
	# python run.py -constraint accuracy -bound 0.8 -approach baseline 
	# python run.py -constraint cost -bound 300 -approach baseline 

	'''
	Configurations
	'''
	parser = argparse.ArgumentParser(description = 'Description')
	parser.add_argument('-a', help='factor for power law distribution',default=5, type=int)
	parser.add_argument('-synthetic','--synthetic',action='store_true',help="it is synthetic")
	parser.add_argument('-record','--record',action='store_true',help="record the results")
	parser.add_argument('-record-test','--record-test',action='store_true',help="record the results")
	parser.add_argument('-order','--order',action='store_true',help="Optimizer considering selectivity and order")
	parser.add_argument('-balance','--balance',action='store_true',help="Optimizer considering trade-off")
	parser.add_argument('-n', help='Number of tasks, enter it', default=40, type=int)
	parser.add_argument('-nquery', help='Number of queries, enter it', default=100, type=int)
	parser.add_argument('-qdist', help='query distribution', default='uniform', type=str)
	parser.add_argument('-mdist', help='model repository distribution', default='uniform', type=str)
	parser.add_argument('-flag', help='high accuracy model is costly', default=1, type=int)
	parser.add_argument('-constraint', help='Type of constraint, enter accuracy/cost', default='cost', type=str)
	parser.add_argument('-approach', help='Type of approach, enter optimizer/baseline', default='optimizer', type=str)
	parser.add_argument('-bound', help='constraint bound, enter it',default=40, type=float)
	args = parser.parse_args()

	
	time = run(args)
