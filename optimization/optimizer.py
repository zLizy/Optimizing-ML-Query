## installation:
## https://www.gurobi.com/documentation/9.0/quickstart_windows/ins_the_anaconda_python_di.html
import gurobipy as gp
from gurobipy import GRB
import numpy as np

class Optimizer(object):
	def __init__(self,query,M,T,Accuracy,cost,constraint,bound):
		nModels = len(M)
		nTasks= len(T)

		Cost = [[cost[i] if Accuracy[i,j] !=0 else 5000 for j in range(nTasks)] for i in range(nModels)]

		performance_dict = {}
		cost_model_dict = {}
		for i,m in enumerate(M):
			cost_model_dict[m] = cost[i]
			for j,t in enumerate(T):
				performance_dict[(m,t)] = [Accuracy[i,j], Cost[i][j]]

		# key, values (accuracy, cost)
		Tuples, *performance = gp.multidict(performance_dict)

		# Models, *performance = gp.multidict({
		#     'M1': [0.96,0,0,0, 15], # car
		#     'M2': [0.98,0,0,0, 30], # car
		#     'M3': [0,0.93,0,0, 20], # bus
		#     'M4': [0,0.95,0,0, 40], # bus
		#     'M5': [0,0,0.96,0.98,5], # red & yellow
		#     'M6': [0,0,0.96,0.97,10]#red & yellow
		# })

		tupleCost = performance[-1]
		tupleAccu = performance[0]


		model=gp.Model('RAPs_complex')
		model.Params.LogToConsole = 0

		self.CostMax = 5000
		self.query = query
		self.Cost = Cost
		self.constraint = constraint
		self.bound = bound
		self.cost_model_dict = cost_model_dict
		self.M = M
		self.T = T
		self.Tuples = Tuples
		self.tupleCost = tupleCost
		self.tupleAccu = tupleAccu
		self.model = model		

	def printResult(self,v,ms,v1):
		if self.model.status == GRB.OPTIMAL:
			model_task_list = {}
			print('Optimal objective: %g' % self.model.objVal)
			 # Compute total matching score from assignment variables
			# print("Selected models:")
			for m, t in self.Tuples:
				if v[m, t].x > 1e-6:
			# 		print(v[m, t].varName, v[m, t].x)
					model_task_list[t]=m
			# print(model_task_list)
			# print("----------------")
			
			if self.constraint == 'cost':
				total_matching_score = 0
				for m in self.M:
					if ms[m].x > 1e-6:
						total_matching_score += self.cost_model_dict[m]*ms[m].x
				# print('cost:',total_matching_score)
				# if model_task_list == {}:
					# print('list: ',model_task_list)
				return model_task_list, list(model_task_list.keys()),self.model.objVal, total_matching_score
			else:
				# print(v1.varName,v1.getAttr(GRB.Attr.X))
				return model_task_list, list(model_task_list.keys()),v1.getAttr(GRB.Attr.X), self.model.objVal


			# print('Total matching score: ', total_matching_score)  

			# print("#################")
			# print("#################")
			

		elif self.model.status == GRB.INF_OR_UNBD:
			print('Model is infeasible or unbounded')
			return {},[],0,0
		elif self.model.status == GRB.UNBOUNDED:
			print('Model is unbounded')
			return {},[],0,0
		else:
			print('Optimization ended with status %d' % self.model.status)
			return {},[],0,0


	def getSteps(self,step):
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


	def recordInitialExpr(self,v):
		# record the expression for each step
		stepMap = {}
		for i, t in enumerate(self.T):
			expr = gp.QuadExpr()
			expr.add(v.prod(self.tupleAccu,'*',t))
			v1 = self.model.addVar(0,1,1,vtype=GRB.CONTINUOUS, name=t)
			self.model.addConstr(v1==expr, name='constraint.'+t)
			stepMap[t] = v1

		# print(stepMap.keys())
		return stepMap

	def recordIntermediateResult(self,v):

		stepMap = self.recordInitialExpr(v)
		steps = self.query.split('\n')[:-1]
		for step in steps:
			self.model.update()
			name, operator, objects = self.getSteps(step)
			# print(name,operator,objects)

			if operator == '&':
				expr = stepMap[objects[0]]*stepMap[objects[1]]
				v1 = self.model.addVar(0,1,1,vtype=GRB.CONTINUOUS, name=name)
				self.model.addConstr(v1==expr, name='constraint.'+name)
				stepMap[name] = v1
			else:
				expr = stepMap[objects[0]] + stepMap[objects[1]] - stepMap[objects[0]]*stepMap[objects[1]]
				v1 = self.model.addVar(0,1,1,vtype=GRB.CONTINUOUS, name=name)
				self.model.addConstr(v1==expr, name='constraint.'+name)
				stepMap[name] = v1

		return v1

	def gurobiOptimizer(self):

		# Decision variables
		# combination assignment
		v = self.model.addVars(self.Tuples,vtype=GRB.BINARY, name="assign")
		# model assignment
		ms = self.model.addVars(self.M,vtype=GRB.BINARY, name='modelSelectivity')

		# Task constraints
		self.model.addConstrs((v.sum('*',t)==1 for t in self.T),name='Task')

		# Cost constraints / Binary constraints
		l = 0
		u = len(self.T)
		e = 0.1 #1.0e-5
		self.model.addConstrs((v.sum(m,'*') <= 1-e+(u-1+e)*ms[m] for m in self.M), name='upper')
		self.model.addConstrs((v.sum(m,'*') >= (1-l-e)*ms[m]+l for m in self.M),name='lower')
		self.model.addConstr((v.prod(self.tupleCost))<= self.CostMax, name='bound')
		
		# Compute Accuracy in each step
		v1 = self.recordIntermediateResult(v)

		# Objective
		if self.constraint == 'cost':
			self.model.addConstr((ms.prod(self.cost_model_dict) <= self.bound), name=self.constraint)
			self.model.setObjective(v1,GRB.MAXIMIZE)
		else:
			self.model.addConstr((v1 >= self.bound), name=self.constraint)
			self.model.setObjective((ms.prod(self.cost_model_dict)),GRB.MINIMIZE)

		# Run optimization engine
		self.model.params.NonConvex = 2
		self.model.optimize()

		assignment,order,_A,_C = self.printResult(v,ms,v1)

		return assignment,order,_A,_C

	def optimize(self):
		return self.gurobiOptimizer()


if __name__ == '__main__':
	# Model and Task sets
	# M1, M2 are color model; M3, M4 are object model
	M = ['M1','M2','M3','M4','M5','M6']
	T = ['car','bus','red','yellow']
	

	Accuracy = np.array([[0.96,0,0,0],
						[0.98,0,0,0],
						[0,0.93,0,0],
						[0,0.95,0,0],
						[0,0,0.96,0.98],
						[0,0,0.96,0.97]])

	cost = [15,30,20,40,5,10]

	query = 'car&red,s0\nbus&yellow,s1\ns0|s1,s2'

	
	# Objective Constraint
	CostBound = 200
	AccuBound = 0.90
	
	optimizer = Optimizer(query,M,T,Accuracy,cost,CostBound)
	res = optimizer.optimize()
