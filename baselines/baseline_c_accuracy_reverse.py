'''
Definition of baseline 2: 
	when accuracy is constrained, select the cheapest model each time, until it satisfies the conditions

'''
import numpy as np
import timeit

def getSteps(step):
		# print(step)
		expr, name = step.split(',')
		if '&' in expr:
			objects = expr.split('&')
			operator = '&'
		else:
			objects = expr.split('|')
			operator = '|'
		return name, operator, objects
	    # [s0, &, [car,red]

	    # Accuracy = np.array([[0.96,0,0,0],
					# 	[0.98,0,0,0],
					# 	[0,0.93,0,0],
					# 	[0,0.95,0,0],
					# 	[0,0,0.96,0.98],
					#   [0,0,0.96,0.97]])

def reverse(selected_model,c,duplicate):
	if not duplicate:
		selected_model['cost']-= c
	selected_model['model'].pop()
	return selected_model

def getBaseline2(query,M,T,Cost,Accuracy,bound,start,selected_model={'min_cost':0},min_accu=0,flag=True,level=0,duplicate=False,second=False):

	# print('T',T)
	if timeit.default_timer() - start > 1 and not second:
		return False,0,0,{}
	if 'model' not in selected_model.keys():
		selected_model['model'] = []
		selected_model['cost'] = 0
		selected_model['min_cost'] = 500
		selected_model['accuracy'] = 0

	t_ind = 0
	task = T[0]
	# if True:
	# 	print(T)
	# 	print(level,task)
	# print(Accuracy.shape)
	# sorted_arg_model = list(np.argsort(np.divide(Accuracy[:,t_ind],Cost[:,t_ind])))[::-1]# descending order, the first index is the largest
	sorted_arg_model = list(np.argsort(Accuracy[:,t_ind]))[::-1]
	sorted_arg_model = sorted_arg_model[:len(np.nonzero(Accuracy[:,t_ind])[0])]
	# print(len(sorted_arg_model))
	for x,m_ind in enumerate(sorted_arg_model):
		# if level ==0:
		# 	print('level 0:',x,M[m_ind])

		if Cost[m_ind,t_ind] == 500:
			return False,0,0,{}

		# if a task can solve multiple tasks, then remove the tasks)
		answered_tasks_ind = []
		answered_tasks = []
		for i,a in enumerate(Accuracy[m_ind,:]):
			if a != 0:
				answered_tasks_ind.append(i)
				answered_tasks.append(T[i])
		# print('answered_tasks',answered_tasks_ind,T,answered_tasks, Accuracy[m_ind,:])

		_T = [t for i,t in enumerate(T) if i != t_ind]

		
		# co-solved tasks or accuracy==0
		# for ind,t in enumerate(answered_tasks_ind):
		# 	ta = answered_tasks[ind]
		# 	if ta not in selected_model.keys():
		# 		selected_model[ta] = [M[m_ind],Accuracy[m_ind,t],Cost[m_ind,t]]
			# elif selected_model[ta][1] < Accuracy[m_ind,t]:
			# 	selected_model[ta] = [M[m_ind],Accuracy[m_ind,t],Cost[m_ind,t]]
		selected_model[task] = [M[m_ind],Accuracy[m_ind,t_ind],Cost[m_ind,t_ind]]
		
		if selected_model['model'] == []:
			selected_model['cost'] += Cost[m_ind,t_ind]
		elif M[m_ind] not in list(zip(*selected_model['model']))[1]:
			selected_model['cost'] += Cost[m_ind,t_ind]
		else:
			duplicate = True
		selected_model['model'].append((task,M[m_ind]))
		# print(selected_model)
			# check if the new model can beat the previous one
			
		
		# if selected_model[task][0] == M[m_ind]:
		# 	continue

		if _T == []:
			
			steps = query.split('\n')[:-1]
			for step in steps:
				name, operator, objects = getSteps(step)
				if operator == '&':
					accu = selected_model[objects[0]][1]*selected_model[objects[1]][1]
				else:
					accu = selected_model[objects[0]][1] + selected_model[objects[1]][1] - selected_model[objects[0]][1]*selected_model[objects[1]][1]
				selected_model[name] = [name,accu,0]
			# if accu < bound:
			# 	print('accu',accu)
			if accu>=bound:
				if selected_model['cost'] < selected_model['min_cost']:
					# print('change cost')
					selected_model['min_cost'] = selected_model['cost']
					selected_model['best_model'] = selected_model['model'].copy()
					selected_model['accuracy'] = accu
					# print('changed',selected_model)
				selected_model = reverse(selected_model,Cost[m_ind,t_ind],duplicate)
				duplicate = False
				# print('reverse',selected_model)
				# print('continue')
				continue
			elif 'accuracy' not in selected_model.keys():
				return False,0,0,{}
			else:
				# print(selected_model['accuracy'], selected_model['min_cost'],{item[0]:item[1] for item in selected_model['model']})
				return True, selected_model['accuracy'], selected_model['min_cost'],{item[0]:item[1] for item in selected_model['model']}
		
		# new_t_ind = [i for i in range(len(T)) if i not in answered_tasks_ind]
		# _M = [m for i,m in enumerate(M) if i != m_ind]
		_M = M
		# _Accuracy = np.delete(Accuracy, m_ind, 0)
		# _Cost = np.delete(Cost, m_ind, 0)
		_Accuracy = np.delete(Accuracy, t_ind, 1)
		_Cost = np.delete(Cost, t_ind, 1)
		# _Accuracy = _Accuracy[:,new_t_ind]
		# _Cost = _Cost[:,new_t_ind]

		flag, _A, _C, selected_model = getBaseline2(query,_M,_T,_Cost,_Accuracy,bound,start,selected_model,level=level+1)

		if flag:
			# _model[task] = selected_model[task][0]
			# if selected_model[task][0] not in _model.values():
			# 	_C+=selected_model[task][2]
			# print('return')
			return flag,_A,_C,selected_model
		elif _A == 0:
			return flag, _A, _C, selected_model
			# return flag,_A,_C,_model
		else:
			# if x != len(sorted_arg_model)-1 and t_ind !=len(T)-1:
			selected_model = reverse(selected_model,Cost[m_ind,t_ind],duplicate)
			duplicate = False
			# flag = True
			continue

	# print('loop all')
	return False, selected_model['accuracy'], selected_model['min_cost'],selected_model


if __name__ == '__main__':

	M = ['M1','M2','M3','M4','M5','M6']
	T = ['car','bus','red','yellow']
	

	Accuracy = np.array([[0.96,0,0,0],
						[0.98,0,0,0],
						[0,0.93,0,0],
						[0,0.95,0,0],
						[0,0,0.96,0.98],
						[0,0,0.96,0.97]])

	cost = [15,30,20,40,5,10]

	Cost = np.array([[cost[i] if Accuracy[i,j] !=0 else 500 for j in range(len(T))] for i in range(len(M))])
	bound = 0.9

	query = 'car&red,s0\nbus&yellow,s1\ns0|s1,s2\n'

	flag,_A, _C,plan = getBaseline2(query,M,T,Cost,Accuracy,bound)
	print(flag,_A, _C,plan)