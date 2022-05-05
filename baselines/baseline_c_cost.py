'''
Definition of baseline 1: 
	when cost is constrained, select the most accurate model each time, until it satisfies the conditions

'''
import numpy as np

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


def getBaseline1(query,M,T,Cost,Accuracy,bound,selected_model={},flag=True):
	
	bestplan = []
	# for t_ind,task in enumerate(T):
	task = T[0]
	t_ind = 0
	# m_ind,t_ind = np.unravel_index(np.argmax(Accuracy[:,i]), a.shape)
		
	# sorted_arg_model = list(np.argsort(np.divide(Accuracy[:,t_ind],Cost[:,t_ind])))[::-1] # descending order, the first index is the max
	sorted_arg_model = list(np.argsort(Accuracy[:,t_ind]))[::-1]

	# print(Cost)
	for m_ind in sorted_arg_model:
		if Accuracy[m_ind,t_ind] == 0.0:
			return False,0,0,{}

		if bound-Cost[m_ind,t_ind]<0:
			continue
		
		# if a task can solve multiple tasks, then remove the tasks
		answered_tasks_ind = []
		answered_tasks = []
		for i,a in enumerate(Accuracy[m_ind,:]):
			if a != 0:
				answered_tasks_ind.append(i)
				answered_tasks.append(T[i])

		for ind,t in enumerate(answered_tasks_ind):
			selected_model[answered_tasks[ind]] = Accuracy[m_ind,t]

		_T = [t for i,t in enumerate(T) if t not in answered_tasks]
		
		if _T == []:
			steps = query.split('\n')[:-1]
			for step in steps:
				name, operator, objects = getSteps(step)
				if operator == '&':
					accu = selected_model[objects[0]]*selected_model[objects[1]]
				else:
					accu = selected_model[objects[0]] + selected_model[objects[1]] - selected_model[objects[0]]*selected_model[objects[1]]
				selected_model[name] = accu

			if accu==0:
				continue
			else:
				return True,accu,Cost[m_ind,t_ind],{t:M[m_ind] for t in answered_tasks}
		
		new_t_ind = [i for i in range(len(T)) if i not in answered_tasks_ind]
		_M = [m for i,m in enumerate(M) if i != m_ind]
		_Accuracy = np.delete(Accuracy, m_ind, 0)
		_Cost = np.delete(Cost, m_ind, 0)
		_Accuracy = _Accuracy[:,new_t_ind]
		_Cost = _Cost[:,new_t_ind]

		# print('cost',Cost[m_ind,t_ind])
		flag, _A,_C, _model = getBaseline1(query,_M,_T,_Cost,_Accuracy,bound-Cost[m_ind,t_ind],selected_model)
		if flag:
			# bestplan.append((answered_tasks,M[m_ind]))
			# bestplan+=_model #(red,model1)
			for t in answered_tasks:
				_model[t] = M[m_ind]
			_C+=Cost[m_ind,t_ind]
			return flag,_A,_C, _model
		else:
			flag = True
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
	Cost = np.array([[cost[i] if Accuracy[i,j] !=0 else 5000 for j in range(len(T))] for i in range(len(M))])
	bound = 40
	query = 'car&red,s0\nbus&yellow,s1\ns0|s1,s2\n'

	flag, _A,_C, _model  = getBaseline1(query,M,T,Cost,Accuracy,bound)
	print(flag, _A,_C, _model )
