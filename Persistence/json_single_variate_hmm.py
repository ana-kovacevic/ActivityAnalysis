
############# SINGLE VARIATE
def hmm_to_dict_single_variate(activity, model):
    mean=model.means_[0][0]
    var=model.covars_[0][0][0]
    trans_mat=model.transmat_
    dict={activity:{'mean':mean, 'var':var, 'trans_mat':trans_mat.tolist()}}
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



























