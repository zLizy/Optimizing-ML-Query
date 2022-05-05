## installation:
## https://www.gurobi.com/documentation/9.0/quickstart_windows/ins_the_anaconda_python_di.html
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from functools import reduce
from util import *

class Optimizer(object):
	def __init__(self,query_idx,query,steps,M,T,Accuracy,cost,selectivity,constraint,bound,query_type='DNF'):
		nModels = len(M)
		nTasks= len(T)

		Accuracy = Accuracy ** 1.5

		Cost = [[cost[i] if Accuracy[i,j] !=0 else 5000 for j in range(nTasks)] for i in range(nModels)]
		cost_model_dict = {}
		for i,m in enumerate(M):
			cost_model_dict[m] = cost[i]

		S_H = selectivity #{t:selectivity[i] for i,t in enumerate(T)} # {task:s}
		S_H_minus = {key:1-val for key,val in S_H.items()} # {task:1-s}

		# Models, *performance = gp.multidict({
		#     'M1': [0.96,0,0,0, 15], # car
		#     'M2': [0.98,0,0,0, 30], # car
		#     'M3': [0,0.93,0,0, 20], # bus
		#     'M4': [0,0.95,0,0, 40], # bus
		#     'M5': [0,0,0.96,0.98,5], # red & yellow
		#     'M6': [0,0,0.96,0.97,10]#red & yellow
		# })


		Tuples, performance_A = self.getTupleDict(M,T,Accuracy)
		tupleAccu = performance_A[0]
		Tuples, performance_C = self.getTupleDict(M,T,Cost)
		tupleCost = performance_C[0]

		Tuples_O, performance_S = self.getTupleDict(T,range(nTasks),[selectivity[t] for t in T])
		tupleSAll = performance_S[0]
		Tuples_O, performance_S_ = self.getTupleDict(T,range(nTasks),[1-selectivity[t] for t in T])
		tupleSAll_ = performance_S_[0]


		Group_G,G = self.getGroup(query,query_type,'G')
		Tuples_G,performance_G = self.getTupleDict(Group_G,range(nTasks),1)
		selectivity_G = []
		for t in G:
			mul = 1
			for key in t:
				mul *= S_H[key]
			selectivity_G.append(mul)
		# selectivity_G = [reduce(lambda x,y: S_H[x]*S_H[y],t) for t in G]

		Group_H,H = self.getGroup(query,query_type,'H')
		Tuples_H,performance_H = self.getTupleDict(Group_H,range(nTasks),1)

		Group_W = Group_G+Group_H
		Tuples_W, performance_W = self.getTupleDict(Group_W,range(nTasks),1)

		Tuples_C, performance_new_r = self.getTupleDict(range(nTasks),M,1)

		Tuples_new_o, performance_new_r = self.getTupleDict(range(nTasks),T,1)
		

		model=gp.Model('RAPs_complex')
		model.Params.LogToConsole = 0

		self.CostMax = 5000
		self.steps = steps
		self.query_idx = query_idx
		self.query = query
		self.Cost = Cost
		self.constraint = constraint
		self.bound = bound
		self.cost_model_dict = cost_model_dict
		self.selectivity_G = selectivity_G
		self.M = M
		self.T = T
		self.Group_G = Group_G
		self.Group_H = Group_H
		self.Group_W = Group_W
		self.G = G
		self.Tuples = Tuples # Assignment
		self.tupleCost = tupleCost
		self.tupleAccu = tupleAccu
		self.tupleSAll = tupleSAll
		self.tupleSAll_ = tupleSAll_
		self.tupleC = Tuples_C
		self.tupleG = Tuples_G
		self.tupleH = Tuples_H
		self.tupleO = Tuples_O
		self.tupleW = Tuples_W
		self.tupleO_new = Tuples_new_o
		self.model = model		


	def gurobiOptimizer(self):

		TLen = len(self.T)
		# Decision variables
		#
		# Assignment variables
		v = self.model.addVars(self.Tuples,lb=0,ub=1,vtype=GRB.BINARY, name="assign")
		
		# model assignment
		# ms = self.model.addVars(self.M,vtype=GRB.BINARY, name='modelSelectivity')
		
		# Order variables
		o = self.model.addVars(self.tupleO,lb=0,ub=1,vtype=GRB.BINARY, name="order") # order

		# G
		g = self.model.addVars(self.tupleG,lb=0,ub=1,vtype=GRB.BINARY, name="g") # group

		# H
		h = self.model.addVars(self.tupleH, lb=0,ub=1,vtype=GRB.CONTINUOUS, name="h") # order
		
		# W
		w = self.model.addVars(self.tupleW, lb=0,ub=1, vtype=GRB.CONTINUOUS, name="w") # order

		# R
		r = self.model.addVars(self.tupleW, lb=0,ub=1,vtype=GRB.CONTINUOUS, name="r") # record multiplications

		# C
		c = self.model.addVars(self.tupleC,lb=0,vtype=GRB.CONTINUOUS, name="c") # record multiplications

		# new_o
		new_o = self.model.addVars(self.tupleO_new,lb=0,ub=1,vtype=GRB.CONTINUOUS, name="new_o") # record multiplications

		# cost of models
		m_c = self.model.addVars(self.M,lb=0,vtype=GRB.CONTINUOUS,name='Model_Cost')

		self.model.update()

		# Constraints
		#
		# Task constraints
		self.model.addConstrs((v.sum('*',t)==1 for t in self.T),name='Task')

		# Cost constraints / Binary constraints
		l = 0
		u = TLen
		e = 0.1 #1.0e-5
		#self.model.addConstrs((v.sum(m,'*') <= 1-e+(u-1+e)*ms[m] for m in self.M), name='upper')
		#self.model.addConstrs((v.sum(m,'*') >= (1-l-e)*ms[m]+l for m in self.M),name='lower')
		self.model.addConstr((v.prod(self.tupleCost))<= self.CostMax, name='bound')
		
		# Order constraint
		self.model.addConstrs((o.sum(t,'*')==1 for t in self.T),name='Task_once')
		self.model.addConstrs((o.sum('*',idx)==1 for idx in range(TLen)),name='Order_once')

		# Add G constraints: if A & B within a group(A,B) is both selected, then g = 0
		for j in range(TLen):
			if j == 0:
				self.model.addConstrs((var == 0 for var in g.select('*',j)),name='g_'+str(j))
			else:
				self.model.addConstrs((var <= o.sum(p,range(j)) for idx,var in enumerate(g.select('*',j)) for p in self.G[idx]),name='g_'+str(j)+'_smaller')
				# self.model.addConstrs((var >= 1-len(self.G[idx])+o.sum(self.G[idx],range(j)) for idx,var in enumerate(g.select('*',j))),name='g_'+str(j)+'_larger')
		
		# h constraints
		for i in range(TLen):
			self.model.addConstrs((var == 1-o.prod(self.tupleSAll_,self.G[gi],i) for gi,var in enumerate(h.select(self.Group_H,i))),name='h'+str(i))
		# new_h
		new_h = self.model.addVars(self.tupleH, lb=0,ub=1,obj=1, vtype=GRB.CONTINUOUS, name="new_h") # order
		# new_h constraints
		for i in range(TLen):
			if i == 0:
				self.model.addConstrs((new_h[gh,i] == 1 for gh in self.Group_H),name='new_H'+str(i))
			else:
				self.model.addConstrs((new_h[gh,i] == new_h[gh,i-1] * h[gh,i-1] for gh in self.Group_H), name='new_H'+str(i))
		
		# w constraints
		for idx in range(TLen):
			# variable G
			self.model.addConstrs((w[gg,idx] == 1-g[gg,idx]*self.selectivity_G[i] for i,gg in enumerate(self.Group_G)),name='w_g'+str(idx))
			self.model.addConstrs((w[gh,idx] == 1-o.sum(self.G[i],idx)*(1-new_h[gh,idx]) for i,gh in enumerate(self.Group_H)),name='w_h'+str(idx))
		
		# r constraints
		for gi,gw in enumerate(self.Group_W):
			if gi == 0:
				self.model.addConstrs((r[gw,i] == w[gw,i] for i in range(TLen)),name='r'+gw)
			else:
				self.model.addConstrs((r[gw,i] == r[self.Group_W[gi-1],i]*w[gw,i] for i in range(TLen)),name='r'+gw)

		# new_o: order x task
		for i in range(TLen):
			self.model.addConstrs((new_o[i,t] == r[self.Group_W[-1],i]*o[t,i] for t in self.T),name='new_o_'+str(i))


		# c constraints
		for i in range(TLen):
		    for m in self.M:
		        expr = gp.QuadExpr()
		        for t in self.T:
		            expr += new_o[i,t]*v[m,t]
		#         print(expr)
		        self.model.addConstr((c[i,m] == expr*self.cost_model_dict[m]),name='c_'+str(i)+'_'+m)

		self.model.update()


		# Compute Accuracy of the query
		v1 = self.recordIntermediateResult(v)

		# Compute Cost of the query plan
		# deduplicate model cost, only counting the first present cost
		self.model.addConstrs((m_c[m] == gp.max_(c.select('*',m)) for m in self.M),name='m_c')
		m_cost = m_c.sum()

		# Objective
		# model without execution order
		if self.constraint == 'cost':
			# self.model.addConstr((ms.prod(self.cost_model_dict) <= self.bound), name=self.constraint)
			self.model.addConstr((m_cost<=self.bound),name=self.constraint)
			self.model.setObjective(v1,GRB.MAXIMIZE)
		else:
			self.model.addConstr((v1 >= self.bound), name=self.constraint)
			# self.model.setObjective((ms.prod(self.cost_model_dict)),GRB.MINIMIZE)
			self.model.setObjective((m_cost),GRB.MINIMIZE)

		# Run optimization engine
		self.model.params.NonConvex = 2
		self.model.optimize()

		if self.model.status == GRB.OPTIMAL:
			assignment,order,_A,_C = self.printResult(v,v1,o,new_o,c,m_c,m_cost)
		elif self.model.status == GRB.INFEASIBLE:
			print('Model is infeasible')
			print(self.model.computeIIS())
			self.model.computeIIS()
			self.model.write("iis/model"+str(self.query_idx)+".ilp")
			return {},[],0,0

		return assignment,order,_A,_C

	def optimize(self):
		return self.gurobiOptimizer()


	def getSteps(self,query):
		return getSteps(query)

	def getGroup(self,query,query_type,v_name):
		opr = '|'
		opr_gr = '&'
		if query_type == 'CNF':
			opr = '&'
			opr_gr = '|'

		group = query.split(opr)
		G = []
		Group_G = []
		for i,x in enumerate(group):
			G.append(x.replace(' ','').replace('(','').replace(')','').split(opr_gr))
			Group_G.append(v_name+str(i))
		# G0 = ['red','car']
		# G1 = ['yellow','bus']
		# G = [G0,G1]
		# Group_G = ['G0','G1']
		return Group_G,G


	def getTupleDict(self,P,Q,V):
		performance_dict = {}
		V = np.asarray(V)
		l = len(V.shape)
		for i,p in enumerate(P):
			for j,q in enumerate(Q):
				if l == 0:
					performance_dict[(p,q)] = V
				elif l == 1:
					performance_dict[(p,q)] = [V[i]]
				else:
					performance_dict[(p,q)] = [V[i][j]]
		# print(performance_dict)
		Tuples, *performance = gp.multidict(performance_dict)
		return Tuples,performance

	def printVariables(self,Tuples,v):
		for i, j in Tuples:
			if v[i, j].x > 1e-6:
				print(v[i, j].varName, v[i, j].x) 

	def printFlatVariables(self,Tuple,v):
		for t in Tuple:
			if v[t].x != 1e-6:
				print(v[t].varName,v[t].x)

	def printResult(self,v,v1,o,new_o,c,m_c,m_cost):
		if self.model.status == GRB.OPTIMAL:
			model_task_list = {}
			print('Optimal objective: %g' % self.model.objVal)
			 # Compute total matching score from assignment variables
			for m, t in self.Tuples:
				if v[m, t].x > 1e-6:
					model_task_list[t]=m

			order_dict = {}
			for t,i in self.tupleO:
				if o[t,i].x > 1e-6:
					order_dict[t] = i
			# sort and print predicate order
			new_order = sorted(order_dict.items(), key=lambda x: x[1], reverse=True)
			pre_order = [item[0] for item in new_order]
			# print(pre_order)

			# print('======== o  ========')
			# self.printVariables(self.tupleO,o)
			# print('======== new_o  ========')
			# self.printVariables(self.tupleO_new,new_o)
			# print('======== c  ========')
			# self.printVariables(self.tupleC,c)
			# print('======== m_c ========')
			# self.printFlatVariables(self.M,m_c)

			
			if self.constraint == 'cost':
				return model_task_list, pre_order,self.model.objVal, m_cost.getValue()
			else:
				# print(v1.varName,v1.getAttr(GRB.Attr.X))
				# print(model_task_list)
				return model_task_list, pre_order,v1.getAttr(GRB.Attr.X), self.model.objVal

			# print('Total matching score: ', total_matching_score)  
		
		# print status
		elif self.model.status == GRB.INFEASIBLE:
			print('Model is infeasible')
			model.computeIIS()
			model.write("iis/model"+str(self.query_idx)+".ilp")
			return {},0,0
		elif self.model.status == GRB.INF_OR_UNBD:
			print('Model is infeasible or unbounded')
			return {},0,0
		elif self.model.status == GRB.UNBOUNDED:
			print('Model is unbounded')
			return {},0,0
		else:
			print('Optimization ended with status %d' % self.model.status)
			return {},0,0


	def getComponents(self,step):
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
		steps = self.steps.split('\n')[:-1]
		for step in steps:
			self.model.update()
			name, operator, objects = self.getComponents(step)
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

	selectivity = [0.2,0.3,0.4,0.5]

	steps = 'car&red,s0\nbus&yellow,s1\ns0|s1,s2'
	query_ori = 'car & red | yellow & bus'

	# Objective Constraint
	constraint = 'cost'
	CostBound = 200
	AccuBound = 0.90
	
	optimizer = Optimizer(query_ori,steps,M,T,Accuracy,cost,selectivity,constraint,CostBound)
	res = optimizer.optimize()
