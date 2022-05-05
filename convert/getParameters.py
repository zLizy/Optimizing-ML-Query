import re
import os.path
import random
import pandas as pd
from sympy.parsing.sympy_parser import parse_expr
from sympy import *
# from optimization.optimizer import *


def getDF(path):
	df = pd.read_csv(path,index_col=0)
	return df

def getPredicates(Bxp):
	predicates = Bxp.atoms()
	# print(predicates)
	if predicates=={True} or predicates=={False}:
		return []
	return predicates

def buildPath(assignment,Bxp,df_config,atom,answered,Etype,flag=True):

	plan = []
	# visited.append()
	if list(getPredicates(Bxp)) != []:
		p = list(getPredicates(Bxp))[0]
		
		if atom == []:
			atom.append((p,))
		
		e_parent = p.name # e=['x1']
		model = assignment[e_parent]
		
		if model not in answered:
			answered.append(model)
			# write script
			idx = int(re.search( r'\d+', model).group())
			extendScript(df_config[df_config['index']==idx],atom,Etype)

		# p = True
		# print(atom)
		e_plus = buildPath(assignment,Bxp.subs({p:True}),df_config,atom+[(p,'>')],answered,Etype,flag=True)

		# p = False
		# print(atom)
		e_minus = buildPath(assignment,Bxp.subs({p:False}),df_config,atom+[(p,'=')],answered,Etype,flag=False)
		
		if plan == []:
			plan = [e_parent, e_plus, e_minus]


	return plan

def extendScript(row,atom,Etype):
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
	s = "{sqlflow}\n{select}{where}\nTO RUN hebafer/yolov3-sqlflow:latest\n"
	s += "CMD \"yolov3_detect_variant.py\",\n\"--dataset=coco\",\n\"--model={model}\",\n\"--latency={latency}\",\n\"--lag={lag}\",\n\"--tasks={tasks}\"\nINTO result;\n"

	# if flag == True: no filter condition
	if len(atom) == 1:
		sqlflow = 'DROP TABLE IF EXISTS coco.result;'
		where = ''
		select = 'SELECT * FROM coco.images'
	else:
		# print('atom',atom[1:])
		sqlflow = ''
		select = 'SELECT * FROM coco.result\n'
		where = 'WHERE '
		# print(atom)
		for (a,flag) in atom[1:]:
			where += f'{a}{flag}0 AND '
		where = where[:-5]

	model = row['model'].values[0]
	latency = row['latency'].values[0]
	lag = row['lag'].values[0]
	tasks =','.join([item for item in row['tasks'].values[0].replace('[','').replace(']','').replace('\n','').split(' ') if item != ''])
	# print(s.format(where=where,latency=latency,lag=lag))
	
	with open('script/script_'+Etype+'.sql','a') as f:
		f.write(s.format(sqlflow=sqlflow,select=select,where=where,model=model,latency=latency,lag=lag,tasks=tasks))
		f.write('\n')

	# return s.format(sqlflow=sqlflow,where=where,latency=latency,lag=lag)


def writeScript(assignment,query,Etype):

	if not assignment:
		return 0
	Bxp = parse_expr(query)
	config_path = 'model_config_task.csv'
	df_config = getDF(config_path)

	plan = buildPath(assignment,Bxp,df_config,[],[],Etype,True)
	print(plan)


if __name__ == '__main__':
	query = 'bus | person | handbag & truck & clock & chair | tennis_racket | sink'
	assignment = {'tennis_racket': 'model_164', 'truck': 'model_164', 'bus': 'model_105', 'handbag': 'model_105', 'person': 'model_154', 'clock': 'model_154', 'chair': 'model_154', 'sink': 'model_154'}
	
	writeScript(assignment,query,Etype='opt')


