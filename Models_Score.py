import pandas as pd
from Persistence import pickle_hmm
import json
import data_preparation as dp
import numpy as np


'''
###############################
SINGLE VARIATE
##############################
'''

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
    '''
    :param users_activities: 
    :return: 
    '''
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



def get_users_ges_activities(users_ges_activities_models):
    '''
    :param users_ges_activities_models: 
    :return: 
    '''
    dict={}
    for user, ges_activities_models in users_ges_activities_models.items():
        dict_ges={}
        for ges, activity_models in ges_activities_models.items():
            activities=list(activity_models.keys())
            activities.remove('transmat')
            activities.remove('covars')
            dict_ges.update({ges:activities})
        dict.update({int(user):dict_ges})
    return dict


def predict_multi_variate(data, users_ges_activities):
    '''
    :param data: 
    :param users_ges_activities: 
    :return: 
    '''
    for user, ges_activities in users_ges_activities.items():
        for ges, activities in ges_activities.items():
            model=pickle_hmm.load_pickle_hmm_multi_variate(user, ges)
            prep_data=dp.prepare_data(data, user, activities)
            clusters=model.predict(prep_data.iloc[:,2:])
            probas=model.predict_proba(prep_data.iloc[:,2:])
            probas_np=np.array(probas)
            max_probas=np.amax(probas_np,1)
            prep_data['cluster']=clusters
            prep_data['max_probability']=max_probas
            #a=pd.melt(prep_data, id_vars=['user_in_role_id', 'interval_end','cluster', 'max_probability'], value_vars=activity)
            prep_data.to_csv('Data/clustered_data/multi_variate_clusters/citizen_id_'+str(user)+'_'+ges+'.csv')
    return 0

json_users_activities_models=open('Models/HMM/JSON/multi_variate_hmms.json').read()
users_activities_models=json.loads(json_users_activities_models)
users_ges_activities=get_users_ges_activities(users_activities_models)
a=predict_multi_variate(data, users_ges_activities)











