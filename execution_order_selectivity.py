import argparse
from sympy import *
from sympy.parsing.sympy_parser import parse_expr

# from TDSIM import BuildPlans,getPredicates


def getName(p,e,branch):
	# Xe = visited
	# Xp = [item for item in [p.name] if item not in Xe] # [p] - Xe
	if e == 'scan':
		return p.name
	elif branch == True:
		return p.name
		# e=p.name+'+'
	else:
		return p.name
		# e[p.name]['-'] = {}
		# e.append(p.name+'-')
	# return e


def getPredicates(Bxp):
	predicates = Bxp.atoms()
	# print(predicates)
	if predicates=={True} or predicates=={False}:
		return []
	return predicates

def TDSIM(e, Bxp, Asg, branch):

	if Asg in [Memo.keys()]:
		return Memo[Asg]

	bestcost = 0
	bestplan = []
	# visited.append()
	for p in getPredicates(Bxp):
		global count
		count = count + 1
		e_parent = getName(p,e,branch) # e=['x1']
		# p = True
		e_plus, c_plus = TDSIM(e_parent, Bxp.subs({p:True}),Asg+(p.name+'+',),True)
		# p = False
		e_minus, c_minus = TDSIM(e_parent, Bxp.subs({p:False}),Asg+(p.name+'-',),False)
		cost = cost_dict[p.name] \
			+ selectivity_dict[p.name]*c_plus \
			+ (1-selectivity_dict[p.name])*c_minus
		if bestplan == [] or bestcost < -(cost):
			bestplan = [e_parent, e_plus, e_minus]
			bestcost = -cost

	Memo[Asg] = bestplan
	# print(Asg) 

	return Memo[Asg], -bestcost


def buildPath(Bxp,branch):

	plan = []
	# visited.append()
	if list(getPredicates(Bxp)) != []:
		p = list(getPredicates(Bxp))[0]
	
		global count
		count = count + 1
		e_parent = p.name # e=['x1']
		# p = True
		e_plus = buildPath(Bxp.subs({p:True}),True)
		print('e_plus',e_plus)
		# p = False
		e_minus = buildPath(Bxp.subs({p:False}),False)
		print('e_minus',e_minus)
		
		if plan == []:
			plan = [e_parent, e_plus, e_minus] 

	return plan

def run(args,selectivity=False,random=False):

	query = args.query.replace("object=","").replace("color=","").replace(";","")
	query = query.replace("and","&").replace("or","|")
	print(query)
	Bxp = parse_expr(query)
	cost = 0
	# selectivity
	if selectivity:
		plan,cost = TDSIM(e,Bxp,Asg,branch)
	if random:
		# random order
		plan = buildPath(Bxp,branch)
	print(plan,cost,count)

	return


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description = 'Description')
	parser.add_argument('--query', default='car & red | bus & yellow',help='your query, enter it',type=str)
	args = parser.parse_args()


	e='scans'
	Asg=tuple(['scan'])
	branch=True
	global Memo
	Memo={}
	count=0
	cost_dict = {'red':4,'yellow':5,'car':10, 'bus':15}
	selectivity_dict = {'red':0.7,'yellow':0.6,'car':0.5,'bus':0.4}

	# query1 = '(x1 & x2) | (x3 & x4)'

	run(args,random=True)