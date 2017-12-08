import pandas as pd
import data_preparation as dp
from HMM_Optimization import HMM_optimization as hmm_opt
import model_persistance as mp


'''
READ DATA, SELECT USERS AND GES/ACTIVITIES
'''
data = pd.read_csv('Data/activities_out.csv') # Reads data with original and normalized values for each user and activity
users=[66,67]
ges_activities=dp.get_dict_ges_activities() # Add weights as a list to dictionary

'''
LEARN OPTIMAL MODELS
'''
optimal_multi_variate=hmm_opt.get_optimal_hmms_for_users_multi_variate(data=data, users=users, cov_type='diag')
optimal_single_variate=hmm_opt.get_optimal_hmms_for_users_single_variate(data=data, users=users, cov_type='diag')

'''
WRITE MODELS TO PICKLE
'''

mp.write_hmms_to_pickle_single_variate(optimal_hmms_single_variate=optimal_single_variate)
mp.write_hmms_to_pickle_multi_variate(optimal_hmms_multi_variate=optimal_multi_variate)

'''
WRITE MODELS TO JSON
'''
mp.user_dict_singlevariate_JSON(optimal_single_variate)


def create_multivariate_dict_for_JSON(users_ges_activities_models):
    user_dict={}
    for user, ges_activities_models in users_ges_activities_models.items():

    dict.update({user:user_dict})


def create_dict_user_level_multi_variate(ges_activities_models):
    dict={}
    for ges, activities_models in ges_activities_models.items():
        for activities, model in activities_models:
            dict.update(create_dict_for_node_hmm_JSON_multi_variate(activities, model))
        dict.update({ges:dict})
    return dict




def create_dict_for_node_hmm_JSON_multi_variate(activities, model):
    means = model['model'].means_
    activities = model['activities']
    covars = model['model'].covars_
    dict={}
    for z in zip(activities, means, covars):
        print(z[1])
        dict.update({z[0]:{'means':list(z[1]), 'covars':list(z[2])}})
    return dict
















