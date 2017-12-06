'''
MODEL PERSISTANCE JSON
'''

def create_dict_activities_means_covars(user, activities, model):
    '''    
    :param model: HMM model
    :return: 
    '''
    means = model.means_
    covars = model.covars_
    transmat = model.transmat_

    dict = {}
    for ac in zip (activities, means, covars):
        dict.update({ac[0]:{'means':means[0].tolist(),'covars':covars[0].tolist()}})
    dict.update({'transmat': transmat})
    dict={user:dict}

    return dict

'''
def persist_model_JSON(model, path):
    import json
    with open('JSONdata_multi.json', 'w') as outfile:
        json.dump(dict_users, outfile)

'''


########## Write mdodel in pickle
def persist_pickle_hmm(model, path, filename):
    '''
    :param model: 
    :param path: 
    :param filename: 
    :return: 
    '''
    from sklearn.externals import joblib
    whole_path=path+filename
    joblib.dump(model, path+'/'+filename+'.pkl')
    joblib.load("filename.pkl")
######## Load model from pickle
def load_pickle_hmm(path, filename):
    '''
    :param path: 
    :param filename: 
    :return: model 
    '''
    from sklearn.externals import joblib
    whole_path = path + filename
    model=joblib.load(path + '/' + filename + '.pkl')
    return model









##### Model persist JSON

'''
dict4json_single={'user': user, 'Activity': 'sleep_deep_time', 'means':model.means_.tolist(), 'covars':model.covars_.tolist(), 'transmat':model.transmat_.tolist()}

dict4json_multi={'user': {'id':user, 'Activities': activities, 'means':model.means_.tolist(), 'covars':model.covars_.tolist(), 'transmat':model.transmat_.tolist()}}

'''