import pickle
import pandas as pd

# my_dict = { 'Ali': 9, 'Sid': 1, 'Luna': 7, 'Sim': 12, 'Pooja': 4, 'Jen': 2}
# with open('data.json', 'wb+') as fp:
#     pickle.dump(my_dict, fp)

with open('model_repository.json', 'rb+') as fp:
    data = pickle.load(fp)
print(data)
print(type(data))

s_dict = {}
m_dict = {}

metrics = 'recall'
for key, value in data.items():
    name = 'model_'+str(key)
    m_dict[name] = {value['predicate']:value[metrics]}
    s_dict[name] = {value['predicate']:round(1-value['reduction_rate'],4)}

df_model = pd.DataFrame.from_dict(m_dict,orient='index')
df_model['cost'] = 21
print(df_model)
# df_model.to_csv('coco_2014_model_stats_'+metrics+'.csv')

df_selectivity = pd.DataFrame.from_dict(s_dict,orient='index')
df_selectivity.to_csv('coco_2014_selectivity.csv')

df_stat = pd.DataFrame.from_dict(data,orient='index')
# df_stat.to_csv('../config/model_config_coco_2014.csv')
