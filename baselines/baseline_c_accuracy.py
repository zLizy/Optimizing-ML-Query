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


def getBaseline2(query,M,T,Cost,Accuracy,bound,start,selected_model={},flag=True,level=0):

	# if timeit.default_timer() - start > 15:
	# 	return False,0,0,{}

	t_ind = 0
	task = T[0]
	# if level <=1:
	# 	print(T)
	# 	print(level,task)
	# m_ind,t_ind = np.unravel_index(np.argmax(Accuracy[:,i]), a.shape)
	sorted_arg_model = list(np.argsort(Cost[:,t_ind])) # asscending order, the first index is the cheapest
	
	# sorted_arg_model = list(np.argsort(np.divide(Accuracy[:,t_ind],Cost[:,t_ind])))[::-1] # descending order, the first index is the largest

	# sorted_arg_model = list(np.argsort(Accuracy[:,t_ind]))[::-1] # descending order, the first index is the largest
	sorted_arg_model = sorted_arg_model[:len(np.nonzero(Accuracy[:,t_ind])[0])]

	for x,m_ind in enumerate(sorted_arg_model):
		# if level == 0 :
			# print('level',level,x,M[m_ind])

		if Cost[m_ind,t_ind] == 500:
			return False,0,0,{}

		if Accuracy[m_ind,t_ind] < 0.3:
			continue

		# if a task can solve multiple tasks, then remove the tasks)
		answered_tasks_ind = []
		answered_tasks = []
		for i,a in enumerate(Accuracy[m_ind,:]):
			if a != 0:
				answered_tasks_ind.append(i)
				answered_tasks.append(T[i])
		# print('answered_tasks',answered_tasks_ind,T,answered_tasks, Accuracy[m_ind,:])

		_T = [t for i,t in enumerate(T) if t not in answered_tasks]

		# co-solved tasks or accuracy==0
		for ind,t in enumerate(answered_tasks_ind):
			selected_model[answered_tasks[ind]] = Accuracy[m_ind,t]
		
		if _T == []:
			steps = query.split('\n')[:-1]
			for step in steps:
				name, operator, objects = getSteps(step)
				if operator == '&':
					accu = selected_model[objects[0]]*selected_model[objects[1]]
				else:
					accu = selected_model[objects[0]] + selected_model[objects[1]] - selected_model[objects[0]]*selected_model[objects[1]]
				selected_model[name] = accu
			if accu<bound:
				# print('accu',accu)
				continue
			else:
				# print(accu, Cost[m_ind,t_ind])
				return True, accu, Cost[m_ind,t_ind], {t:M[m_ind] for t in answered_tasks}
		
		new_t_ind = [i for i in range(len(T)) if i not in answered_tasks_ind]
		_M = [m for i,m in enumerate(M) if i != m_ind]
		_Accuracy = np.delete(Accuracy, m_ind, 0)
		_Cost = np.delete(Cost, m_ind, 0)
		_Accuracy = _Accuracy[:,new_t_ind]
		_Cost = _Cost[:,new_t_ind]

		flag, _A, _C, _model = getBaseline2(query,_M,_T,_Cost,_Accuracy,bound,start,selected_model,level=level+1)

		if flag:
			for t in answered_tasks:
				_model[t] = M[m_ind]
			_C+=Cost[m_ind,t_ind]
			return flag,_A,_C,_model
		else:
			# flag = True
			continue

	
	return False,0,0,{}


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