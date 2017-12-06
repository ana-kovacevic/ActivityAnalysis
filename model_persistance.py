'''
MODEL PERSISTANCE JSON
'''

import pickle
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