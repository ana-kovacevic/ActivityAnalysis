
'''
MODEL PERSISTANCE JSON
'''

#### SINGLE VARIATE

def create_dict_for_node_hmm_JSON_single_variate(activity, model):
    '''    
    :param model: HMM model
    :return:
     Creates dict of the Hmm model for persistance in JSON
    '''
    means = model.means_
    covars = model.covars_
    transmat = model.transmat_
    dict = {'mean': means, 'covar': covars, 'transmat': transmat}
    dict = {activity: dict}
    return dict


def create_dict_node_user_level_single_variate(user, activities_models):
    '''
    ### Creates node (dictionary) for single user and all activities

    :param user: citizen_id
    :param activities_models: tuple of activity names and corresponding models 
    :return: 
    '''
    dict = {}
    for activity, model in activities_models:
        dict = create_dict_for_node_hmm_JSON_single_variate(activity, model)
        dict.update(dict)
    dict = {user: dict}
    return (dict)




############# SINGLE VARIATE
def hmm_to_dict_single_variate(activity, model):
    mean=model.means_[0]
    var=model.covars_[0]
    trans_mat=model.transmat_
    dict={activity:{'mean':mean, 'var':var, 'trans_mat':trans_mat}}
    return dict

def user_dict_singlevariate_JSON(users_activities_models):
    users_dict={}
    for user, activities in users_activities_models.items():
        activities_dict={}
        for activity, model in activities.items():
            model=hmm_to_dict_single_variate(activity, model)
            activities_dict.update(model)
        users_dict.update({user:activities_dict})
    return users_dict



























