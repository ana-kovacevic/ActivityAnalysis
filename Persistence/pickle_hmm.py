########## Write mdodel in pickle

import pickle

def create_path_single_variate(user, activity):
    path = 'Models/HMM/Pickle/citizen_id_' + str(user) + '/'
    file_name='citizen_id_'+str(user)+'_activity_'+activity
    extension='.pkl'
    whole_path=path+file_name+extension
    return whole_path

def create_path_multi_variate(user, activity):
    path = 'Models/HMM/Pickle/citizen_id_' + str(user) + '/'
    file_name='citizen_id_'+str(user)+'_ges_'+activity
    extension='.pkl'
    whole_path=path+file_name+extension
    return whole_path



def persist_pickle_hmm(model, path):
    '''
    :param model: 
    :param path: 
    
    :return:
     Persists both multi and single variate models
    '''

    #filehandler = open(str(whole_path), 'w')
    file=open(path, "wb")
    pickle.dump(model, file)
    #joblib.dump(model, whole_path+'.pkl')



def load_pickle_hmm_single_variate(user, activity):
    path=create_path_single_variate(user, activity)
    file = open(path, "rb")
    model=pickle.load(file)
    return model


def load_pickle_hmm_multi_variate(user, activity):
    path=create_path_multi_variate(user, activity)
    file = open(path, "rb")
    model=pickle.load(file)
    return model



def write_hmms_to_pickle_single_variate(optimal_hmms_single_variate):
    '''
    :param:  
    :return: 
    '''
    for user, activity_models in optimal_hmms_single_variate.items():
        for activity, model in activity_models.items():
            path = create_path_single_variate(user, activity)
            persist_pickle_hmm(model, path)

####### PICKLE

def write_hmms_to_pickle_multi_variate(optimal_hmms_multi_variate):
    '''
    :param optimal_hmms_single_variate: tuple containing citizen_id, subfactor and optimal hmm (by # of clusters) 
    :return: 
    '''
    for user, subfactors_models in optimal_hmms_multi_variate.items():
        for subfactor, models in subfactors_models.items():
            model=models['model']
            path = create_path_multi_variate(user, subfactor)
            persist_pickle_hmm(model, path)

