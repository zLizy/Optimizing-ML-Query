# log
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/test')
from test.my_logger import get_logger

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


logger = get_logger(__name__, 'logs/query_optimization_pareto_voc_accuracy.log', use_formatter=False)


def run(args):


	countModel = 0
	countPareto = 0
	tempList = []
	tempQuery = []

	logger.info('================')
	logger.info('======== {} ========'.format(args.outdir))

	if args.synthetic:
		logger.info('synthetic: {}'.format(args.synthetic))

		# get model repository
		df = getRandomRepository(args.mdist,args.a)
		tasks = ['P'+str(idx) for idx in range(len(df.columns)-1)]
		df.columns = tasks+['cost']
		
		logger.info('len of df is: '+str(len(df.columns)))
		logger.info('columns (tasks): {}'.format(df.columns))

		models = ['M'+str(idx) for idx in range(len(df.index))]
		df.index = models

		logger.info('rows (models): {}'.format(df.index))

		# get Pareto summary
		df_pareto = getParetoSummary(df,args.mdist,args.a)
		# logger.info('-----------')
		# logger.info('df_pareto (head):')
		# logger.info('\t'+ df_pareto.head().to_string().replace('\n', '\n\t')) 
		# logger.info('-----------')
		
		# synthesize queries
		df_query = getQueries(len(tasks),args.qdist,args.qfile)
		# queryList = getQueries(len(tasks),args.qdist)
	else:
		df = getYOLORepository(args.repo)
		config = 'convert/'+args.data_type+'_model_config_'+ '_'.join(args.repo.split('_')[3:])
		# df = df.drop(['model_44'])
		# df = df.drop(['model_30'])
		logger.info('configuration file: {}'.format(config))

		tasks = [c for c in df.columns if c != 'cost']
		models = df.index
		logger.info('len of df is: '+str(len(df.columns)))

		# get Pareto summary
		df_pareto = getParetoSummary(df,synthetic=args.synthetic)
	
		# synthesize queries
		df_query = getQueries(query_type=args.qtype,synthetic=args.synthetic,qfile=args.qfile)
		
		data_type = args.data_type

		logger.info('data_type: {}'.format(data_type))
		logger.info('-----------')
		logger.info('df_query:')
		logger.info('\t'+ df_query[['query','#predicate','len','form']].to_string().replace('\n', '\n\t')) 
		logger.info('-----------')
	
	if args.query_idx != -1:
		df_tmp = pd.DataFrame(columns=df_query.columns)
		df_query = df_tmp.append(df_query.iloc[args.query_idx,:])

	# idx = 1
	_time = 0
	data_process_time = 0

	if args.synthetic:
		df = getParetoModelOnly(df,df_pareto,args.mdist,args.a,synthetic=args.synthetic)
		logger.info('Get pareto models only.')
	# get Pareto model only
	elif args.approach == 'baseline':
		df = getParetoModelOnly(df,df_pareto,args.mdist,args.a,synthetic=args.synthetic)
		logger.info('approach: greedy optimizer')
	elif args.order:
		logger.info('approach: order optimizer')
	else:
		logger.info('approach: model {}'.format(args.approach))

	# queryList = queryList[:1] + queryList[2:]#[queryList[0]]#

	
	for idx,row in df_query.iterrows():
		logger.info('~~~~~~ configuration ~~~~~~~')
		if idx<0:
			logger.info('query '+ str(idx))
		else:
			logger.info('query '+ str(idx))
			query = row['query']
			try:
				query_size = str(row['#predicate'])
			except: query_size = '0'
			try:
				query_type = row['form'].lower()
			except:
				query_type = args.qtype

			logger.info('query: {}'.format(query))
			logger.info('query_type: {}'.format(query_type))

			bound = args.bound
			# if args.constraint == 'cost':
				# bound = args.bound * len(T)
			logger.info('constraint: {}'.format(args.constraint))
			logger.info('bound: {}'.format(bound))
			
			# Parse the query and retrieve perdicates and execution steps
			T,steps = getSteps(str(query))

			logger.info('-------')
			logger.info ('T: {}'.format(T))
			logger.info('steps: {}'.format(steps))
			

			# data process
			start = time.time()#timeit.default_timer()
			selected_col = T#[t.replace('_',' ') for t in T]

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

			logger.info('M: {}'.format(M))
			# logger.info('-------')
			# logger.info('cost:')
			# logger.info(cost)
			# logger.info('-------')
			# logger.info('Accuracy:')
			# logger.info(Accuracy)
			# logger.info('-------')

			# selectivity = [0.8] * len(T)
			# selectivity
			if args.order:
				selectivity = getSelectivity(selected_col,row_selected,args.repo,args.synthetic,args.MODEL_SELECTIVITIY,args.data_type)
			end = time.time()#timeit.default_timer()
			data_process_time = end-start

			# compute runtime
			# ## optimizer
			if args.approach == 'optimizer':
				start = time.time()#timeit.default_timer()
				if args.order:
					# logger.info('approach: {} optimizer'.format(args.order))
					# from optimization.optimizer_order import Optimizer
					# check boundary
					if args.check_boundary:
						checkBoundary(idx,str(query),steps,M,T,cost,Accuracy,selectivity=selectivity,constraint=args.constraint,query_type=query_type,type='order',synthetic=args.synthetic,MODEL_SELECTIVITIY=args.MODEL_SELECTIVITIY)
						continue
					else:
						from optimization.optimizer_order_all import Optimizer
						optimizer = Optimizer(idx,str(query),steps,M,T,Accuracy,cost,selectivity,args.constraint,bound,query_type,MODEL_SELECTIVITIY=args.MODEL_SELECTIVITIY)
						assignment,pre_order,_A,_C = optimizer.optimize()
						model_assignment = model_order = conditions = task_selectivity = {}
						if assignment != {}:
							model_assignment, model_order, conditions,task_selectivity = assignment
						end = time.time()#timeit.default_timer()
						_time = end-start
						logger.info('Accu: {}, Cost: {}'.format(round(_A,3),round(_C,2)))
						logger.info('time: {}'.format(_time))
						logger.info('assignment: {}'.format(model_assignment))
						logger.info('model_order: {}'.format(model_order))
						logger.info('conditions: {}'.format(conditions))
						logger.info('task_selectivity')
						logger.info(task_selectivity)
						logger.info('pre_order: {}'.format(pre_order))
						# with open('assignment.txt', 'a') as f:
						# 	f.write('readme')

				else:
					from optimization.optimizer import Optimizer
					optimizer = Optimizer(steps,M,T,Accuracy,cost,args.constraint,bound)
					# assignment = {task:model}
					assignment,pre_order,_A,_C = optimizer.optimize()
					end = time.time()#timeit.default_timer()
					_time = end-start
					logger.info('Accuracy: {}, Cost: {}'.format(_A,_C))
					logger.info('time: {}'.format(_time))
					logger.info(assignment)
				
				if not args.synthetic:
					Etype = 'basic'
					if args.record_test:
						Etype = 'test_'+Etype
					path = writeScript(data_type,assignment,str(query),idx,pre_order,Etype,args.constraint,args.bound,query_type,query_size,args.order,outdir=args.outdir,script_folder=args.scriptdir,config=config)
				if args.record:
					writeIntermediateParetoSummary(args,idx,df_pareto,query,T,assignment,_A,_C,_time,data_process_time,outdir=args.outdir)
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

					# check boundary
					if args.check_boundary:
						checkBoundary(idx,str(query),steps,M,T,Cost,Accuracy,args.constraint,synthetic=args.synthetic)
					else:
						# run
						flag,_A,_C,assignment = getBaseline1(steps,M,T,Cost,Accuracy,bound,selected_model={})
						end = time.time() #timeit.default_timer()
						_time = end-start
						logger.info('flag:, {}'.format(flag))
						logger.info('_A'.format(_A))
						logger.info('_C'.format(_C))
						logger.info('assignment: {}'.format(assignment))
					
						if args.order:
							pre_order = []
						else:
							pre_order = list(assignment.keys())

						if not args.synthetic:
							# print()
							Etype = 'baseline'
							if args.record_test:
								Etype = 'test_'+Etype
							path  = writeScript(data_type,assignment,str(query),idx,pre_order,Etype,args.constraint,args.bound,query_type,query_size,args.order,outdir=args.outdir,script_folder=args.scriptdir,config=config)
						if args.record:
							writeIntermediateParetoSummary(args,idx,df_pareto,query,T,assignment,_A,_C,_time,data_process_time,approach='baseline_pareto',outdir=args.outdir)
				else:
					## baseline2
					start = time.time()  # timeit.default_timer()
					from baselines.baseline_c_accuracy import getBaseline2
					if args.check_boundary:
						checkBoundary(idx,str(query),steps,M,T,Cost,Accuracy,constraint=args.constraint,synthetic=args.synthetic)
					else:
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
							writeIntermediateParetoSummary(args,idx,df_pareto,query,T,assignment,_A,_C,_time,data_process_time,approach='baseline_pareto',outdir=args.outdir)

						if not args.synthetic:
							Etype = 'baseline'
							if args.record_test:
								Etype = 'test_'+Etype
							path = writeScript(data_type,assignment,str(query),idx,pre_order,Etype,args.constraint,args.bound,query_type,query_size,args.order,outdir=args.outdir,script_folder=args.scriptdir,config=config)

							logger.info('----- outpath: {} ------'.format(path))
		# idx+=1	

	return _time


if __name__ == '__main__':
	
	# python query_optimizer.py -mdist uniform -qdist uniform -synthetic -record -constraint cost -bound 200
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
	# python3 query_optimizer.py -constraint accuracy -bound 0.85 -synthetic -record
	# python3 query_optimizer.py -constraint cost -bound 50 -record -order -approach baseline
	# python3 query_optimizer.py -constraint accuracy -bound 0.8 -approach baseline 
	# pytho3 query_optimizer.py -constraint cost -bound 300 -approach baseline 

	'''
	Configurations
	'''
	parser = argparse.ArgumentParser(description = 'Description')
	parser.add_argument('-a', help='factor for power law distribution',default=5, type=int)
	parser.add_argument('-synthetic','--synthetic',action='store_true',help="it is synthetic")
	parser.add_argument('-record','--record',action='store_true',help="record the results")
	parser.add_argument('-record-test','--record-test',action='store_true',help="record the results")
	parser.add_argument('-order','--order',action='store_true',help="Optimizer considering selectivity and order")
	parser.add_argument('-repo','--repo',default='model_stats_ap.csv')
	parser.add_argument('-qtype','--qtype',default='dnf')
	parser.add_argument('-qfile','--qfile',default='simulation/coco_query_cnf_gt.csv')
	parser.add_argument('-data','--data-type',default='coco')
	parser.add_argument('-outdir','--outdir',default='coco',help="record the results")
	parser.add_argument('-bound', help='constraint bound, enter it',default=40, type=float)
	parser.add_argument('-query-idx','--query-idx', help='query index',default=-1, type=int)
	parser.add_argument('-scriptdir','--scriptdir',default='script',type=str)
	parser.add_argument('-check','--check-boundary',action='store_true',help="Check the cost constraint boundary of the baseline")
	

	parser.add_argument('-balance','--balance',action='store_true',help="Optimizer considering trade-off")
	parser.add_argument('-n', help='Number of tasks, enter it', default=40, type=int)
	parser.add_argument('-nquery', help='Number of queries, enter it', default=100, type=int)
	parser.add_argument('-qdist', help='query distribution', default='uniform', type=str)
	parser.add_argument('-mdist', help='model repository distribution', default='uniform', type=str)
	parser.add_argument('-flag', help='high accuracy model is costly', default=1, type=int)
	parser.add_argument('-constraint', help='Type of constraint, enter accuracy/cost', default='cost', type=str)
	parser.add_argument('-approach', help='Type of approach, enter optimizer/baseline', default='optimizer', type=str)
	parser.add_argument('-model-sel','--MODEL-SELECTIVITIY',action='store_true',help="Whether to apply selectivity on a model level")
	
	args = parser.parse_args()

	
	time = run(args)
