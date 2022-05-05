import numpy as np

class MaxAccuracy(object):
	def __init__(self,query,M,T,Accuracy,cost,CostBound):
		nModels = len(M)
		nTodels = len(T)

		Cost = [[cost[i] if Accuracy[i,j] !=0 else 500 for j in range(nTodels)] for i in range(nModels)]

		performance_dict = {}
		cost_model_dict = {}
		for i,m in enumerate(M):
			cost_model_dict[m] = cost[i]
			for j,t in enumerate(T):
				performance_dict[(m,t)] = [Accuracy[i,j], Cost[i][j]]


		self.CostMax = 500
		self.query = query
		self.Cost = Cost
		self.CostBound = CostBound
		self.cost_model_dict = cost_model_dict
		self.M = M
		self.T = T
		self.Tuples = Tuples
		self.tupleCost = tupleCost
		self.tupleAccu = tupleAccu

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

	def getMaxAccuracy(self, name, operator, objects):

		# name, operator, objects = self.getSteps(step)
		model, accu, cost = getMaxAccuracy(name)
		accu1 = stepMap[objects[0]]

	def run(self):
		steps = self.query.split('\n')[:-1]
		for step in steps:
			name, operator, objects = self.getSteps(step)

			if operator == '&':
				expr = stepMap[objects[0]]*stepMap[objects[1]]
				v1 = self.model.addVar(0,1,1,vtype=GRB.CONTINUOUS, name=name)
				self.model.addConstr(v1==expr, name='constraint.'+name)
				stepMap[name] = v1
			else:
				expr = stepMap[objects[0]] + stepMap[objects[1]] - stepMap[objects[0]]*stepMap[objects[1]]
				v2 = self.model.addVar(0,1,1,vtype=GRB.CONTINUOUS, name=name)
				self.model.addConstr(v2==expr, name='constraint.'+name)
				stepMap[name] = v2

		return name, stepMap