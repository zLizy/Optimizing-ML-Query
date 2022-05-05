import re
import os.path
import random
import time
import pandas as pd
import numpy as np
from simulation.synthesize_sampling import generateRandomRepository
from sympy.parsing.sympy_parser import parse_expr
from sympy import *
from optimization.parser import *
from simulation.querygenerator import Expression
from simulation.checkPareto import checkPareto
from optimization.optimizer import *


def is_blank(x):
	return not pd.isnull(x)

def getDF(path):
	df = pd.read_csv(path,index_col=0)
	return df

def getParetoSummary(df,mdist='uniform',a=5,synthetic=True,path=''):
	# method = mapFlag(flag)

	if synthetic:
		if mdist == 'power_law':
			path = 'repository/model_pareto_'+mdist+'_a='+str(a)+'.csv'
		else:
			path = 'repository/model_pareto_'+mdist+'.csv'
	if not synthetic:
		if path == '':
			path = 'repository/yolo_pareto.csv'
	# if os.path.isfile(path):
	# 	df_pareto = getDF(path)
	df_pareto = checkPareto(df,path)
	df_pareto.to_csv(path)
	return df_pareto

def getSelectivity(T,index,repo,synthetic,MODEL_SELECTIVITIY,pp=False,data='coco'):
	if synthetic:
		path = 'repository/selectivity.csv'
		if os.path.isfile(path):
			df = getDF(path)
		else:
			df = pd.DataFrame(columns=['class','selectivity'])
			df['class'] = T
			df['selectivity'] = np.random.rand(len(T))
			df.to_csv(path)
		return dict(zip(df['class'], df['selectivity']))
	else:
		if MODEL_SELECTIVITIY:
			if pp:
				path = 'repository/coco_2014_selectivity.csv'
			else:
				addition = ('_').join(repo.split('_')[5:])
				path = 'repository/coco_selectivity_'+ addition
			df = getDF(path)
			df_selected = df[T]
			df_selected.reindex(index)
			df_selected = df_selected.dropna(how='all').fillna(0)
			selectivity = df_selected.to_numpy()
			return selectivity
		else:
			if data == 'coco':
				df = getDF('repository/coco_selectivity.csv')
			else:
				df = getDF('repository/voc_2012_selectivity.csv')
			return dict(zip(df['class'], df['selectivity']))
	


def getRandomRepository(mdist,a):
	'''
	Get synthesized model repository.
	The result is stored in model_repository.csv
	--mflag: uniform / power_law
	'''
		# write index to file
	# filepath = 'repository/index.csv'
	# if not os.path.isfile(filepath):
	# 	df_index = pd.DataFrame()
	# 	df_index['model']=df.index.tolist()
	# 	df_index['new_name'] = models
	# 	df_index.to_csv(filepath)

	if mdist == 'power_law':
		filepath = 'repository/model_repository_'+mdist+'_a='+str(a)+'.csv'
	else:
		filepath = 'repository/model_repository_'+mdist+'.csv'
	if os.path.isfile(filepath):
		print ("Model repository exist")
		df = getDF(filepath)
	else:
		print ("File not exist")
		print('synthesizing model repository')
		df = generateRandomRepository(mdist,a)
	
	return df

def getYOLORepository(repo):
	# return getDF('repository/model_stats_ap.csv')
	return getDF('repository/'+repo)
	# return getDF('repository/model_stats_f1_ignore.csv')
	# return getDF('repository/model_stats_f1_ignore_pareto_half.csv')
	# return getDF('repository/model_stats_f1_half.csv')#('repository/model_stats_f1_pareto.csv')

def getParetoModelOnly(df,df_pareto,mdist='uniform',a=5,synthetic=True,path=''):
	if synthetic:
		if mdist == 'power_law':
			path = 'repository/pareto_only_model_'+mdist+'_a='+str(a)+'.csv'
		else:
			path = 'repository/pareto_only_model_'+mdist+'.csv'
	if not synthetic:
		if path == '':
			path = 'repository/pareto_only_yolo.csv'
	
	# if os.path.isfile(path):
	# 	df_new = getDF(path)
	# 	return df_new
	indexer = df_pareto.applymap(is_blank)
	df_new = pd.DataFrame(columns=df.columns,index=df.index)
	df_new[indexer] = df
	df_new['cost'] = df['cost']
	print('cost' in df_new.columns)
	df_new.to_csv(path)
	return df_new

def getQueries(N=1,qdist='uniform',query_type='dnf',qfile='',synthetic=1):
	'''
	Query generator
	parameter: 
	N - number of tasks
	qdist - distribution:
			uniform
			gaussian
			power_law
			conjunction
			disjunction
	'''
	df = pd.DataFrame()
	if synthetic:
		# filepath = 'repository/query_'+qdist+'.csv'
		# filepath = 'simulation/DNF_query_'+qdist+'_synthetic.csv'
		filepath = 'simulation/synthetic_query_'+qdist+'.csv'
		if os.path.isfile(filepath):
			print ("Query file exist")
			df = getDF(filepath)
			# queryList = list(getDF(filepath)['query'])
		else:
			print ("Synthesizing queries")
			queryList = createQueries(N,qdist)
	else:
		filepath = qfile
		#'simulation/coco_query_'+query_type+'_gt.csv'
		# filepath = 'simulation/coco_query_dnf_gt.csv'
		# filepath = 'simulation/coco_query_size_2_predicate_4_gt.csv'
		# filepath='simulation/query_gt.csv'
		# filepath='simulation/coco_query_gt.csv'
		if os.path.isfile(filepath):
			print ("Query file exist")
			df = getDF(filepath)
			# df = df[df['form']=='dnf']
			# queryList = list(df['query'])
			# queryList = list(df[df['form']=='dnf']['query'])
		# uni_idx = [11,35,37]
		# pow_idx = [2,5,13,15,27,22,26]
		# _idx = [uni_idx,pow_idx]
		# qdist = ['uniform','power_law']
		# for i,dist in enumerate(qdist):
		# 	filepath = 'simulation/query_'+qdist[i]+'_gt.csv'
		# 	if os.path.isfile(filepath):
		# 		print ("Query file exist")
		# 		queryList += list(getDF(filepath)['query'][_idx[i]])

		# dnf_uni_idx = [19,25]
		# dnf_pow_idx = [11,20]
		# _idx = [dnf_uni_idx,dnf_pow_idx]
		# qdist = ['uniform','power_law']
		# for i,dist in enumerate(qdist):
		# 	filepath = 'simulation/query_dnf_'+qdist[i]+'_gt.csv'
		# 	if os.path.isfile(filepath):
		# 		print ("Query file exist")
		# 		queryList += list(getDF(filepath)['query'][_idx[i]])
		# print(queryList)
	return df


def createQueries(N,qdist='uniform'):

	num_exp = 5
	num_iter = 10
	if qdist == 'conjunction':
		conj_pro = 1
	elif qdist == 'disjunction':
		conj_pro = 0
	else:
		conj_pro = 0.5

	queryList = []
	
	df = pd.DataFrame(columns=['query','#predicate','#&','#|','#()','ratio_&','ratio_|'])
	for i in range(1,num_exp+1):
		# factor for number of predicate in a query
		# maximum #predicate is 2^6 = 64
		for j in range(num_iter):
			num_pred = pow(2,i)
			query = str(Expression(N,num_pred,conj_pro,qdist))
			# print(query)
			queryList.append(query)
			num_and = len(query.split('&'))-1
			num_or = len(query.split('|'))-1
			num_par = len(query.split('('))-1
			item = {'query':query,'#predicate':pow(2,i),\
				'#&':num_and,'#|':num_or,'#()':num_par, \
				'ratio_&':round(num_and/(num_pred-1),4), \
				'ratio_|':round(num_or/(num_pred-1),4)}
			df = df.append(item,ignore_index=True)
	df.to_csv('repository/query_'+qdist+'.csv')
	return queryList


def paretoSummary(df,assignment):

	countPareto = 0
	countNP = 0
	pareto = ''
	non_pareto = ''
	model = ''
	for t,m in assignment.items():
		# if m not in df.index: 
		# 	countPareto += 1
		# 	continue
		if df.loc[m,t] == 1:
			countPareto += 1
			pareto += t+'.'+m
			pareto += '|'
		else: 
			countNP += 1
			non_pareto += t+'.'+m
			non_pareto += '|'
		model += t+'.'+m
		model += '|'
	pareto = pareto[:-1]
	non_pareto = non_pareto[:-1]
	model = model[:-1]
	return countPareto,countNP,pareto,non_pareto,model
	

def mapFlag(flag):
	if flag: 
		method = 'max'
	else: 
		method = 'median'
	return method

def createdf(columns):
	df = pd.DataFrame(columns=columns)
	return df

def getSteps(query):
	'''
	Parse query into steps:
	car & red | bus & yellow
	-->
	car&red,s0
	bus&yellow,s1
	s0|s1,s2
	'''
	# parse query into lists of operations
	result = parse(query)
	steps,i = getSequence(result,0,'')
	# parse query into expression
	Bxp = parse_expr(query)
	predicates = [str(p) for p in list(Bxp.atoms())]

	return predicates,steps


def writeResults(args,query_idx,query,T,all_model,pareto,non_pareto,count_p,count_np,_A,_C,_time,data_process_time,approach,outdir,addition):
	
	# if args.mdist == 'power_law':
	# 	filepath = 'output/summary_'+args.constraint+'_'+args.mdist+'_'+ \
	# 		args.qdist+'_'+approach+'_a='+str(args.a)+'.csv'
	
	
	if args.synthetic:
		if not os.path.exists('output/synthetic'):
			os.makedirs('output/synthetic')
		if args.order:
			filepath = 'output/synthetic/summary_order_'+args.constraint+'_'+args.mdist+'_'+args.qdist+'_'+approach+'.csv'
		else:
			filepath = 'output/synthetic/summary_'+args.constraint+'_'+args.mdist+'_'+args.qdist+'_'+approach+'.csv'
	else:
		# addition = '_f1_dnf'
		if not os.path.exists('output/'+outdir):
			os.makedirs('output/'+outdir)

		if args.order:
			filepath = 'output/'+outdir+'/summary_order_'+args.constraint+'_'+approach+'.csv'
		else:
			filepath = 'output/'+outdir+'/summary_'+args.constraint+'_'+approach+'.csv'

			

	# filepath = 'repository/test.csv'
	if os.path.isfile(filepath):
		df_summary = pd.read_csv(filepath,index_col=0)
	else: 
		df_summary = pd.DataFrame(columns=['query_index','qgetSelectivityuery','#predicate','mdist','qdist','constraint','bound','approach','selected_model',
			'pareto_model','#pareto','non_pareto_model','#non_pareto','pareto_ratio','accuracy','cost','optimization_time','data_process_time','addition'])

	# if not all_model:
	# 	df_summary = df_summary.append({'query':query,'#predicate':len(T),'mdist':args.mdist,'qdist':args.qdist,
	# 	'constraint':args.constraint,'bound':args.bound,'approach':approach},ignore_index=True)
	# else:
	df_summary = df_summary.append({'query_index':query_idx,'query':query,'#predicate':len(T),'mdist':args.mdist,'qdist':args.qdist,
		'constraint':args.constraint,'bound':args.bound,'approach':approach,'selected_model':all_model,'pareto_model':pareto,
		'#pareto':count_p,'non_pareto_model':non_pareto,'#non_pareto':count_np,'pareto_ratio':round(count_p/len(T),4),
		'accuracy':_A,'cost':_C,'optimization_time':round(_time,4),'data_process_time':round(data_process_time,4),'addition':addition},ignore_index=True)
	df_summary = df_summary.sort_values(by=['#predicate','query','qdist','mdist','constraint','approach'])
	df_summary.index = range(len(df_summary))
	df_summary.to_csv(filepath)

def writeIntermediateParetoSummary(args,query_idx,df_pareto,query,T,assignment,_A,_C,time,data_process_time,approach='optimizer',outdir='',addition=[]):
	if assignment != {}:  #{t:m}
		# countModel += len(assignment)
		if args.order:
			assign = assignment[0]
			assignment = list(assignment)
		else:
			assign = assignment
		count_p, count_np, pareto, non_pareto, all_model = paretoSummary(df_pareto, assign)
		# countPareto += count_p
	# if all_model == '':
	# 	tempList.append(non_pareto)
	# 	tempQuery.append(str(non_pareto))
	else:
		all_model=pareto=non_pareto=''
		count_p=count_np=0

	# addition['order_dict'] = assignment[1]
	# addition['condition'] = assignment[2]
	if args.record:
		writeResults(args,query_idx,query,T,assignment,pareto,non_pareto,count_p, count_np,_A,_C,time,data_process_time,approach,outdir,addition)
	else:
		s = '_A,_C,time,data_process_time,approach'.replace(',','\t')
		print(s)
		print(_A,_C,time,data_process_time,approach)
	# return countModel

def writeSummary(args,countModel,countPareto):

	path = 'repository/model_pareto_summary.csv'
	if os.path.isfile(path):
		df_summary = getDF(path)
	else:
		df_summary = createdf(['#predicates','#pareto','ratio','mdist','qdist'])
	
	if countModel:
		df_summary = df_summary.append(\
		{'#predicates':int(countModel),'#pareto':int(countPareto),\
		'ratio':round(countPareto/countModel,4),'mdist':args.mdist,'qdist':args.qdist},\
		ignore_index=True)
	else:
		df_summary = df_summary.append(\
		{'#predicates':int(countModel),'#pareto':int(countPareto),\
		'ratio':0,'mdist':args.mdist,'qdist':args.qdist},\
		ignore_index=True)
	df_summary = df_summary.sort_values(by=['mdist','qdist','#predicates'])
	df_summary.to_csv(path)
	return

def writeQuery(tempList,tempQuery,mdist,qdist):
	# print(tempList)
	# print(tempQuery)
	path = 'repository/non_pareto_query_summary.csv'
	if os.path.isfile(path):
		df_summary = getDF(path)
	else:
		df_summary = createdf(['query','predicate','mdist','qdist'])
	df_temp = createdf(['query','predicate','flag'])
	df_temp['query'] = tempQuery
	df_temp['predicate'] = tempList
	df_temp['mdist'] = [mdist]*len(tempQuery)
	df_temp['qdist'] = [qdist]*len(tempQuery)
	df_summary = df_summary.append(df_temp)
	df_summary.index = range(len(df_summary))
	# df_summary = df_summary.sort_values(by=['M','N','#query'])
	df_summary.to_csv(path)
	return

def getPredicates(Bxp):
	predicates = Bxp.atoms()
	# print(predicates)
	if predicates=={True} or predicates=={False}:
		return []
	return predicates

def buildPath(data_type,assignment,Bxp,order,df_config,atom,parent_answered,path,flag=True):

	import re
	# print(Bxp)
	plan = []
	
	# if Bxp == True or Bxp == False:
	# 	return []
	predicates = list(getPredicates(Bxp))

	if predicates != []:
	# if order != []:
		# p = list(getPredicates(Bxp))[0]
		p = [symbols(o) for o in order if symbols(o) in predicates][-1]
		
		if atom == []:
			atom.append((p,))
		
		e_parent = p.name # e=['x1']
		model = assignment[e_parent]

		# print('model',model,'parent_answered',parent_answered)
		
		if model not in parent_answered:
			i = int(re.findall( r'\d+', model)[0]) #.group())
			# extendScript(df_config[df_config['index']==i],atom,path)
			extendScript(data_type,i,atom,list(assignment.keys()),path)

		# print(str(p)+'=True',Bxp.subs({p:True}))
		# print('order',order)
		
		# p = True
		e_plus = buildPath(data_type,assignment,Bxp.subs({p:True}),order,df_config,atom+[(p,'>')],parent_answered+[model],path,flag=True)

		# p = False
		# print(str(p)+'=False',Bxp.subs({p:False}))
		# print('order',order)
		e_minus = buildPath(data_type,assignment,Bxp.subs({p:False}),order,df_config,atom+[(p,'=')],parent_answered+[model],path,flag=False)
		

		plan = [e_parent, e_plus, e_minus]


	return plan

def extendScript(data_type,model_index,atom,T,path):
	# DROP TABLE IF EXISTS coco.result;
	# SELECT * FROM coco.`images`
	# ORDER BY `images`.`id`  ASC
	# LIMIT 10
	# TO RUN hebafer/yolov3-sqlflow:latest
	# CMD "yolov3_detect_variant.py",
	#     "--dataset=coco",
	#     "--model=yolov3",
	#     "--latency=0.05",
	#     "--lag=100",
 	#    "--tasks=1,2,3,4,5"
	# INTO result;
	# SELECT * FROM coco.images 
	# ORDER BY images.image_id ASC 
	# LIMIT 10 
	# TO RUN hebafer/sqlflow-experiments:latest 
	# CMD "yolov3_detect_variant.py",
	# "--experiment_index=100" INTO result;

	# if flag == True: no filter condition
	base = path.split('/')[-1][:-3]
	if len(atom) == 1:
		s = 'python3 detect.py --experiment_index {index} --base {base} --data_type {data_type}\n'
		with open(path,'a') as file:
			file.write(s.format(index=str(model_index),base=base))
	else:
		s = 'python3 detect.py --experiment_index {index} --conditions {conditions} --base {base} --data_type {data_type}\n'
		conditions = "'"
		for (a,flag) in atom[1:]:
			conditions += "{a}{flag}0,"
		conditions = conditions[:-1]+"'"
		with open(path,'a') as file:
			file.write(s.format(conditions=conditions,index=str(model_index),base=base,data_type=data_type))

def normalScript(data_type,assignment,path,form,outdir,config):

	import re
	s = "python3 detect.py --experiment_index {index} --base {base} --data_type {data_type} --outdir {out} --config {config}\n"
	s_init = "python3 detect.py --experiment_index {index} --base {base} --data_type {data_type} --outdir {out} --config {config} --init\n"

	models = list(set(assignment.values()))
	T = list(assignment.keys())
	base = path.split('/')[-1][:-3]
	# print(base)
	with open(path,'w') as f:
		for i,model in enumerate(models):
			# model_index = int(re.search( r'\d+', model).group())
			model_index = int(re.findall(r'\d+',model)[0])
			if i == 0:
				f.write(s_init.format(index=str(model_index),base=base,data_type=data_type,out=outdir,config=config))
				continue
			f.write(s.format(index=str(model_index),base=base,data_type=data_type,out=outdir,config=config))

def orderScript(data_type,assignment,path,form,outdir,config):
	import re
	
	s = "python3 detect.py --experiment_index {index} --base {base} --data_type {data_type} --outdir {out} --config {config}\n"
	s_init = "python3 detect.py --experiment_index {index} --base {base} --data_type {data_type} --outdir {out} --config {config} --init\n"
	s_con = "python3 detect.py --experiment_index {index} --base {base} --data_type {data_type} --conditions {conditions} --outdir {out} --config {config}\n"

	# models = list(set(assignment.values()))
	# T = list(assignment.keys())
	# {t:m}
	# {i:t}
	# {i:conditons}
	model_task_list, order_dict, plan, task_selectivity = assignment
	print(plan)
	base = path.split('/')[-1][:-3]
	answered_plan = []

	with open(path,'w') as f:
		for i in range(len(order_dict)):
			t = order_dict[i]
			model = model_task_list[t]
			model_index = int(re.search( r'\d+', model).group())
			if model_index in answered_plan:
				continue
			else:
				answered_plan.append(model_index)
			if i == 0:
				f.write(s_init.format(index=str(model_index),base=base,data_type=data_type,out=outdir,config=config))
				continue
			if plan[i] == '':
				f.write(s.format(index=str(model_index),base=base,data_type=data_type,out=outdir,config=config))
			else:
				f.write(s_con.format(index=str(model_index),base=base,data_type=data_type,conditions='"'+plan[i]+'"',out=outdir,config=config))

def writeScript(data_type,assignment,query,idx,pre_order,Etype,constraint,bound,form,predicate_count,order=False,outdir='',script_folder='',config=''):
	# outdir = '_f1_dnf'
	if not assignment:
		return 0

	# script_folder = 'script_new_model/'
	Bxp = parse_expr(query)
	# config_path = 'repository/model_config_task_half_all.csv'
	# df_config = getDF(config_path)

	if constraint == 'cost':
		bound = int(bound)
	else:
		bound = int(bound*100)
	
	if not os.path.exists(script_folder+'/'+outdir):
		print(script_folder+'/'+outdir)
		os.makedirs(script_folder+'/'+outdir)

	if order:
		name = 'order_'+str(idx)+'_'+form+'_'+predicate_count+'_'+constraint+'_'+str(bound)
		path = script_folder+'/'+outdir+'/script_'+name+'.sh'
	else:
		name = Etype+'_'+str(idx)+'_'+form+'_'+predicate_count+'_'+constraint+'_'+str(bound)
		path = script_folder+'/'+outdir+'/script_'+name+'.sh'

	if order:
		# plan = buildPath(name,assignment,Bxp,pre_order,df_config,[],[],path,flag=True)
		orderScript(data_type,assignment,path,form,outdir,config)
		# print(plan)
	else:
		normalScript(data_type,assignment,path,form,outdir,config)
	return path


def checkBoundary(idx,query,steps,M,T,Cost,Accuracy,query_type='dnf',selectivity=[],constraint='cost',bound_range=range(30,150,10),type='baseline',synthetic=False):

	if constraint != 'cost':
		if type == 'baseline':
			from baselines.baseline_c_accuracy import getBaseline2
			for bound in [0.7,0.75,0.8,0.85,0.9,0.95]:
				start = time.time()
				flag,_A,_C,assignment = getBaseline2(steps,M,T,Cost,Accuracy,bound,start,selected_model={})
				print('bound: ',bound)
				print('assignment',assignment)
		else:
			
			from optimization.optimizer_order_all import Optimizer
			for bound in [0.7,0.75,0.8,0.85,0.9,0.95]:
				optimizer = Optimizer(idx,query,steps,M,T,Accuracy,Cost,selectivity,constraint,bound,query_type)
				assignment,pre_order,_A,_C = optimizer.optimize()
				print('bound: ',bound)
				print('assignment',assignment)
	else:
		if type == 'baseline':
			from baselines.baseline_c_cost import getBaseline1
			if synthetic:
				bound_range = range(0,50,5)
			for bound in bound_range:
				flag,_A,_C,assignment = getBaseline1(steps,M,T,Cost,Accuracy,bound,selected_model={})
				print('bound: ',bound)
				print('assignment',assignment)
		else:
			from optimization.optimizer_order_all import Optimizer
			for bound in bound_range:
				optimizer = Optimizer(idx,query,steps,M,T,Accuracy,Cost,selectivity,constraint,bound,query_type)
				assignment,pre_order,_A,_C = optimizer.optimize()
				print('bound: ',bound)
				print('assignment',assignment)
	



