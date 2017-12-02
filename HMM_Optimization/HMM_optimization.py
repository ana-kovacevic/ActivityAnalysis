from hmmlearn.hmm import GaussianHMM

import numpy as np

import pandas as pd


### Creates single variate HMM models for each activity.
### This method should only be called. It calls preparation methods and returns dictionary with elements for plotting and mapping to states
def create_single_variate_clusters(data, user, activities, activity_extremization, activity_weights):
    import data_preparation as dp
    clusters_activities = {}
    for ac in activities:
        pivoted_data=dp.prepare_data(data, user, [ac])
        model = GaussianHMM(n_components=5, covariance_type="full", n_iter=1000).fit(pivoted_data.iloc[:, 2:])
        hidden_states = model.predict(pivoted_data.iloc[:, 2:])
        extreme=activity_extremization[ac]
        weight=activity_weights[ac]
        clusters_activities.update({ac:{'name': ac, 'model':model, 'clusters':hidden_states, 'values':pivoted_data[ac], 'dates':pivoted_data['interval_end'], 'extremization':extreme, 'weight':weight}})
    return clusters_activities




def optimize_number_of_clusters(data, range_of_clusters):
    '''    
    :param data: 
    :param range_of_clusters: 
    :return: 
    '''

    best_value=np.inf # create for maximization and minimization
    best_model=None
    for n_states in range_of_clusters:
        model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000).fit(data)
        log_likelihood = model.score(data)
        criteria=bic_criteria(data, log_likelihood, model)
        if criteria < best_value:
            best_value, best_model = criteria, model
    return best_value, best_model


def bic_criteria(data, log_likelihood, model):
    '''
    :param data: 
    :param log_likelihood: 
    :param model: 
    :return: 
    '''
    n_features = data.shape(2)  ### here adapt for multi-variate
    n_states=len(model.means_)
    n_params = n_states * (n_states - 1) + 2 * n_features * n_states
    logN = np.log(len(data))
    bic = -2 * log_likelihood + n_params * logN
    return(bic)

def aic_criteria(data, log_likelihood, model):
    '''
    :param data: 
    :param log_likelihood: 
    :param model: 
    :return: 
    '''
    n_features = data.shape(2)  ### here adapt for multi-variate
    n_states=len(model.means_)
    n_params = n_states * (n_states - 1) + 2 * n_features * n_states
    aic = -2 * log_likelihood + n_params
    return(aic)