import pandas as pd
import data_preparation as dp
from HMM_Optimization import HMM_optimization as hmm_opt
from Persistence import json_single_variate_hmm as json_single
from Persistence import json_multi_variate_hmm as json_multi
from Persistence import pickle_hmm as pickle_hmm

'''
READ DATA, SELECT USERS AND GES/ACTIVITIES
'''
data = pd.read_csv('Data/activities_out.csv') # Reads data with original and normalized values for each user and activity
users=[66,67,68]
ges_activities=dp.get_dict_ges_activities() # Add weights as a list to dictionary

'''
LEARN OPTIMAL MODELS
'''
optimal_multi_variate=hmm_opt.get_optimal_hmms_for_users_multi_variate(data=data, users=users, cov_type='diag')
optimal_single_variate=hmm_opt.get_optimal_hmms_for_users_single_variate(data=data, users=users, cov_type='diag')


'''
WRITE MODELS TO JSON
'''
import json

dict_single_variate=json_single.user_dict_singlevariate_JSON(optimal_single_variate)
dict_multi_variate=json_multi.create_dict_users(optimal_multi_variate)


with open('Models/HMM/JSON/multi_variate_hmms.json', 'w') as outfile:
    json.dump(dict_multi_variate, outfile)

with open('Models/HMM/JSON/single_variate_hmms.json', 'w') as outfile:
    json.dump(dict_single_variate, outfile)


'''
WRITE MODELS TO PICKLE
'''

pickle_hmm.write_hmms_to_pickle_single_variate(optimal_hmms_single_variate=optimal_single_variate)
pickle_hmm.write_hmms_to_pickle_multi_variate(optimal_hmms_multi_variate=optimal_multi_variate)




'''
user=66 # selects one user id
activities=['sleep_awake_time','sleep_deep_time', 'sleep_light_time', 'sleep_tosleep_time', 'sleep_wakeup_num']
activity_extremization = {'sleep_light_time':'max', 'sleep_deep_time':'max', 'sleep_awake_time':'min', 'sleep_wakeup_num':'min', 'sleep_tosleep_time':'min'}
activity_weights = {'sleep_light_time':0.1, 'sleep_deep_time':0.3, 'sleep_awake_time':0.1, 'sleep_wakeup_num':0.3, 'sleep_tosleep_time':0.2}

'''














