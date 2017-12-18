import pandas as pd
import data_preparation as dp
from HMM_Optimization import HMM_optimization as hmm_opt
from Persistence import json_single_variate_hmm as json_single
from Persistence import json_multi_variate_hmm as json_multi
from Persistence import pickle_hmm as pickle_hmm
from hmmlearn.hmm import GaussianHMM

'''
READ DATA, SELECT USERS AND GES/ACTIVITIES
'''
data = pd.read_csv('Data/activities_out.csv') # Reads data with original and normalized values for each user and activity
users=[66,67,68]
ges_activities=dp.get_dict_ges_activities() # Add weights as a list to dictionary


'''
LOG EXPERIMENTAL RESULTS
'''
import numpy as np
def log_activity_results(data, users, range_of_clusters, cov_type, single_multi):
    '''
    :param data: prepared data (values of activities by columns)
    :param range_of_clusters: range of best number expected e.g. 2:10
    :return:
     Optimizes number of clusters for single citizen
     This is helper method for get_optimal_hmms methods (they work for more citizens)
    '''
    import pickle
    log_results = []
    subfactor_activities = dp.get_dict_ges_activities()
    for user in users:
        for subfactor, activities in subfactor_activities.items():
            for activity in activities:
                prepared_data = dp.prepare_data(data, user, [activity])
                #log = optimize_number_of_clusters(prepared_data.iloc[:, 2:], list(range(2, 11)), cov_type)
                # pivoted_data = prepare_data(data, user, [ac]) old version for data preparation
                for n_states in range_of_clusters:
                    model = GaussianHMM(n_components=n_states, covariance_type=cov_type, n_iter=1000).fit(prepared_data.iloc[: ,2:])
                    log_likelihood = model.score(prepared_data.iloc[: ,2:])
                    criteria_bic = bic_criteria(data, log_likelihood, model)
                    criteria_aic = aic_criteria(data, log_likelihood, model)
                    aic_bic_dict = {'user': user, 'activity': activity, 'n_states': n_states, 'BIC': criteria_bic,
                                    'AIC': criteria_aic, 'Log_Likelihood':log_likelihood}
                    log_results.append(aic_bic_dict)
                    if single_multi == 'single':
                        path = 'Experimental_Evaluation/Models/user_' + str(user) + 'activity_' + activity + '_n_states_' + str(n_states) + '.pkl'
                    if single_multi == 'multi':
                        path = 'Experimental_Evaluation/Models/user_' + str(user) + 'sub_factor_' + activity + '_n_states_' + str(n_states) + '.pkl'
                    file = open(path, "wb")
                    pickle.dump(model, file)
    if single_multi == 'single':
        log_path = 'Experimental_Evaluation/single_variate_log.csv'
    if single_multi == 'multi':
        log_path = 'Experimental_Evaluation/multi_variate_log.csv'

    log = pd.DataFrame(log_results)
    log.to_csv(log_path)
    return log


log_activity_results(data,users, list(range(2, 11)), 'diag', 'single')

data.columns


prepared_data = dp.prepare_data(data, 66, ['sleep_deep_time'])
prepared_data.columns

prepared_data.head()




def bic_criteria(data, log_likelihood, model):
    '''
    :param data: 
    :param log_likelihood: 
    :param model: 
    :return: 
    '''
    n_features = data.shape[1]  ### here adapt for multi-variate
    n_states=len(model.means_)
    n_params = n_states * (n_states - 1) + 2 * n_features * n_states
    logN = np.log(len(data))
    bic = -2 * log_likelihood + n_params * logN
    return(bic)

def aic_criteria(data, log_likelihood, model):
    '''
    :param data: 
    :param log_likelihood: 
    :param model: 
    :return: 
    '''
    n_features = data.shape[1]  ### here adapt for multi-variate
    n_states=len(model.means_)
    n_params = n_states * (n_states - 1) + 2 * n_features * n_states
    aic = -2 * log_likelihood + n_params
    return(aic)


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










