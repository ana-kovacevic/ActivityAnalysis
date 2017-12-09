import pandas as pd
from Persistence import pickle_hmm
import json
import data_preparation as dp
import numpy as np



def get_users_activities(users_activities_models):
    '''
    :param users_activities_models: 
    :return: 
    '''
    dict={}
    for user, activities_models in users_activities_models.items():
        activity_list=[]
        for activity in activities_models.keys():
            activity_list.append(activity)
        dict.update({int(user):activity_list})
    return dict

users_activities = get_users_activities(users_activities_models)

def predict_single_variate(users_activities):
    df_predictions=pd.DataFrame()
    for user, activities in users_activities.items():
        for activity in activities:
            model=pickle_hmm.load_pickle_hmm_single_variate(user, activity)
            prep_data=dp.prepare_data(data, user, [activity])
            clusters=model.predict(prep_data.iloc[:,2:])
            probas=model.predict_proba(prep_data.iloc[:,2:])
            probas_np=np.array(probas)
            max_probas=np.amax(probas_np,1)
            prep_data['cluster']=clusters
            prep_data['max_probability']=max_probas
            a=pd.melt(prep_data, id_vars=['user_in_role_id', 'interval_end','cluster', 'max_probability'], value_vars=activity)
            df_predictions=df_predictions.append(a)
    return df_predictions


data=pd.read_csv('Data/activities_out.csv')

### Maybe add preparation part
users=[66, 67, 68]

'''
###############################
SINGLE VARIATE
##############################
'''
json_users_activities_models=open('Models/HMM/JSON/single_variate_hmms.json').read()
users_activities_models=json.loads(json_users_activities_models)
predictions=predict_single_variate(users_activities)
predictions=predictions.rename(columns={'variable': 'detection_variable_name'})
clustered_data=pd.merge(data, predictions, how='inner', on= ['user_in_role_id', 'detection_variable_name' ,'interval_end'])
clustered_data.to_csv('Data/clustered_data/single_variate_clusters.csv')

'''
##########################
MULTI VARIATE
##########################
'''

json_users_activities_models=open('Models/HMM/JSON/multi_variate_hmms.json').read()
users_activities_models=json.loads(json_users_activities_models)

user=66
subfactor_activities = dp.get_dict_ges_activities()
subfactor='quality_of_sleep'
activities=subfactor_activities[subfactor]
model=pickle_hmm.load_pickle_hmm_multi_variate(user, subfactor)
prepared_data = dp.prepare_data(data,user,activities)
prepared_data.head()
model.predict(prepared_data.iloc[: ,2:])
model.predict_proba(prepared_data.iloc[: ,2:])

    for user in users:
        for subfactor in subfactor_activities.keys():
                activities=subfactor_activities[subfactor]
                prepared_data = dp.prepare_data(data,user,activities)
                best_value, best_model = optimize_number_of_clusters(prepared_data.iloc[: ,2:], list(range(2,11)), cov_type)
                dict_subfactor.update({subfactor:{'model':best_model, 'activities':activities}})
        dict_user={user:dict_subfactor}
        optimal_hmms_multi_variate.update(dict_user)
    return optimal_hmms_multi_variate


def get_users_ges(users_ges_models):
    '''
    :param users_activities_models: 
    :return: 
    '''
    dict={}
    for user, activities_models in users_activities_models.items():
        activity_list=[]
        for activity in activities_models.keys():
            activity_list.append(activity)
        dict.update({int(user):activity_list})
    return dict



def predict_multi_variate(users_activities):
    df_predictions=pd.DataFrame()
    for user, activities in users_activities.items():
        for activity in activities:
            model=pickle_hmm.load_pickle_hmm_single_variate(user, activity)
            prep_data=dp.prepare_data(data, user, [activity])
            clusters=model.predict(prep_data.iloc[:,2:])
            probas=model.predict_proba(prep_data.iloc[:,2:])
            probas_np=np.array(probas)
            max_probas=np.amax(probas_np,1)
            prep_data['cluster']=clusters
            prep_data['max_probability']=max_probas
            a=pd.melt(prep_data, id_vars=['user_in_role_id', 'interval_end','cluster', 'max_probability'], value_vars=activity)
            df_predictions=df_predictions.append(a)
    return df_predictions





