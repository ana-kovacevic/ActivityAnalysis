import pickle

'''
MODEL PERSISTANCE JSON
'''

########## Write mdodel in pickle
def persist_pickle_hmm(model, path, filename):
    '''
    :param model: 
    :param path: 
    :param filename: 
    :return:
     Persists both multi and single variate models
    '''
    whole_path=path+filename
    #filehandler = open(str(whole_path), 'w')
    file=open(whole_path, "wb")
    pickle.dump(model, file)
    #joblib.dump(model, whole_path+'.pkl')


def create_path(user, activity):
    path = 'Data/citizen_id_' + str(user) + '/'
    file_name='citizen_id_'+str(user)+'_activity_'+activity
    extension='.pkl'
    whole_path=path+file_name+extension
    return whole_path



def load_pickle_hmm_single_variate(user, activity):
    path=create_path(user, activity)
    file = open(path, "rb")
    model=pickle.load(file)
    return model


def write_hmms_to_pickle_single_variate(optimal_hmms_single_variate):
    '''
    :param optimal_hmms_single_variate: tuple containing citizen_id, activity and optimal hmm (by # of clusters) 
    :return: 
    '''
    for best in optimal_hmms_single_variate:
        user=best[0]
        activity=best[1]
        model=best[2]
        path='Data/citizen_id_'+str(user)+'/'
        filename='citizen_id_'+str(user)+'_activity_'+activity+'.pkl'
        persist_pickle_hmm(model, path, filename)



def write_hmms_to_pickle_multi_variate(optimal_hmms_multi_variate):
    '''
    :param optimal_hmms_single_variate: tuple containing citizen_id, subfactor and optimal hmm (by # of clusters) 
    :return: 
    '''
    for best in optimal_hmms_multi_variate:
        user=best[0]
        subfactor=best[1]
        model=best[2]
        path='Data/citizen_id_'+str(user)+'/'
        filename='citizen_id_'+str(user)+'_subfactor_'+subfactor+'.pkl'
        persist_pickle_hmm(model, path, filename)


def create_dict_for_node_hmm_JSON_single_variate(activity, model):
    '''    
    :param model: HMM model
    :return:
     Creates dict of the Hmm model for persistance in JSON
    '''
    means = model.means_
    covars = model.covars_
    transmat = model.transmat_
    dict = {'mean':means, 'covar':covars, 'transmat':transmat}
    dict={activity:dict}
    return dict

def create_dict_node_user_level_single_variate(user, activities_models):
    '''
    ### Creates node (dictionary) for single user and all activities
    
    :param user: citizen_id
    :param activities_models: tuple of activity names and corresponding models 
    :return: 
    '''
    dict={}
    for activity, model in activities_models:
        dict=create_dict_for_node_hmm_JSON_single_variate(activity, model)
        dict.update(dict)
    dict={user:dict}

#def create_dict_for_multiple_users_single_variate():




'''
def create_dict_node_user_level(user, ges_activities, model):
        for ges, activity in ges_activities:
            dict = create_dict_for_node_hmm_JSON_single_variate(activity, model)
        dict = {ges: dict}

    for ac in zip(activities, means, covars):
        dict.update({ac[0]: {'means': means[0].tolist(), 'covars': covars[0].tolist()}})
    dict.update({subfactor: dict})


def create_dict_for_node_JSON(user, activities, model):
    
    means = model.means_
    covars = model.covars_
    transmat = model.transmat_

    dict = {}

        for ac in zip (activities, means, covars):
            dict.update({ac[0]:{'means':means[0].tolist(),'covars':covars[0].tolist()}})
        dict.update({subfactor:dict})
    dict.update({'transmat': transmat})
    dict={user:dict}

    return dict

def_create_dict_for

for subfactor, activities in subfactor_activities:

'''









'''
def persist_model_JSON(model, path):
    import json
    with open('JSONdata_multi.json', 'w') as outfile:
        json.dump(dict_users, outfile)

'''










######### Write models in Dictionary (JSON)












##### Model persist JSON

'''
dict4json_single={'user': user, 'Activity': 'sleep_deep_time', 'means':model.means_.tolist(), 'covars':model.covars_.tolist(), 'transmat':model.transmat_.tolist()}

dict4json_multi={'user': {'id':user, 'Activities': activities, 'means':model.means_.tolist(), 'covars':model.covars_.tolist(), 'transmat':model.transmat_.tolist()}}

'''