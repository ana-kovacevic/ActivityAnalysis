'''
################################
MULTI VARIATE
################################
'''


import json
from Persistence import paths





def create_dict_users(users_ges_activities_models):
    dict = {}
    for user, ges_activities_models in users_ges_activities_models.items():
        dict_help = create_dict_ges(ges_activities_models)
        dict.update({user: dict_help})

    return dict


def create_dict_ges(ges_activities_models):
    dict = {}
    for ges, activities_models in ges_activities_models.items():
        activities = activities_models['activities']
        model = activities_models['model']
        dict_activities = create_dict_activities(activities, model)
        dict_activities.update({'covars': model.covars_})
        dict_activities.update({'transmat': model.transmat_})
        dict.update({ges: dict_activities})
    return dict


def create_dict_activities(activities, model):
    means = model.means_
    covars = model.covars_
    dict = {}
    for activity in activities:
        dict.update({activity: {'means': means}})
        # dict.update({activity: {'means': means, 'covars': covars}})
    return dict








