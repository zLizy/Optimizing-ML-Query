import numpy as np
import pandas as pd
import random
import warnings
warnings.filterwarnings('ignore')

def generateRandomRepository(mdist,a):
	df = pd.read_csv('./repository/summary_all_flat.csv',header=0,index_col=0)
	df_accu = df.iloc[:,:-1]
	cost = df['cost']
	cost_dict = dict(cost)

	X=64
	cost_range = 0.4
	# accu_range = 0.1

	vocabularies = df_accu.columns
	model_dict = {i:key for i, key in enumerate(cost_dict.keys())}
	print('##############')
	new_dict = {}
	cost_std = pd.Series(cost_dict).std()
	cost_min = 1 #pd.Series(cost_dict).min()
	for index, row in df_accu.iterrows():
	    model_base = index
	    row = row.dropna()
	    _max = row.max()
	    _min = row.min()
	    mu, sigma = row.mean(),row.std()
	    _classes = row.index
	    for x in range(1,X):
	        model_name = model_base+'_'+str(x)
	        # selected classes
	        total = len(row)
	        # power law selection
	        if mdist == 'power_law':
	        	chosen_idx = total*np.random.power(a, size=np.random.randint(1,total)).astype(int)
	        	selected = [c for i, c in enumerate(_classes) if i in chosen_idx]
	        else:
	        	selected = random.sample(list(_classes), np.random.randint(1,total))
	        _accu = np.random.normal(loc=mu,scale=sigma,size=len(selected))
	        cost_dict[model_name] = np.random.uniform(low=cost_min,high=cost_dict[model_base])
	        new_dict[model_name] = {item: max(min(_accu[i],1),0) for i,item in enumerate(selected)}


	for k,v in new_dict.items():
	    _series = pd.Series(v)
	    _series.name = k
	    df_accu = df_accu.append(_series)
	df_accu['cost'] = pd.Series(cost_dict)

	df_accu = dropEmptyColumns(df_accu)

	if mdist == 'power_law':
		df_accu.to_csv('./repository/model_repository_power_law_a='+str(a)+'.csv')
	else:
		df_accu.to_csv('./repository/model_repository_'+mdist+'.csv')

	return df_accu


def dropEmptyColumns(df):
    for c in df.columns[:-1]:
        data = df.loc[:,[c]]
        data = data.dropna()
        if data.empty:
            print(c)
            df = df.drop(columns=[c])
    return df
