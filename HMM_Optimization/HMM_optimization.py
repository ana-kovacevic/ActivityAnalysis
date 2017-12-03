from hmmlearn.hmm import GaussianHMM
import numpy as np
#import data_preparation as dp
import pandas as pd


### Creates single variate HMM models for each activity.
### This method should only be called. It calls preparation methods and returns dictionary with elements for plotting and mapping to states
def create_single_variate_clusters(data, user, activities, activity_extremization, activity_weights):
    clusters_activities = {}
    for ac in activities:
        pivoted_data=prepare_data(data, user, [ac])
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


####### PREPARATION METHODS ... JUST TEMPORARILY HERE

### Maybe add this to data exploration
def get_users_activities(data, user):
    '''
    :param data: data 
    :param user: user elderly
    :return: all activities

    Gets all activities for specified user and returns activity names and counts
    Exploratory method for selection of users/activities for modelling
    '''
    user_data = data[data['user_in_role_id'] == user]
    d = user_data.groupby(['user_in_role_id', 'detection_variable_name'])['measure_value'].count()
    # d.rename(columns={'measure_value':'count_measure_value'}, inplace=True)
    d = pd.DataFrame(d)
    return d


def select_pivot_users_activities(data, user, activities):
    '''
    Pivots multivariate data - each activity becomes column
    Unnecessary step for single variate time series - maybe remove and adjust prepare data method
    '''
    user_data = data[data['user_in_role_id'] == user]
    user_data = user_data[user_data['detection_variable_name'].isin(activities)]
    pivot_data = user_data.pivot_table(index=['user_in_role_id', 'interval_end'], columns='detection_variable_name',
                                       values='Normalised')
    return pivot_data


def prepare_data(data, user, activities):
    '''
    :param data: transaction data
    :param user: user_in_role_id in integer format
    :param activities: list of activity names 
    :return: 
    '''
    '''
    Takes pivoted data and transforms it in regular DataFrame. 
    Converts dates to date format (for plotting) 
    Sorts data based on dates in order to preserve temporal order
    !!!If used for Single-Variate clustering list have to be passed (for one activity)
    '''
    pivoted_data = select_pivot_users_activities(data, user, activities)
    pivoted_data = pivoted_data.reset_index()
    pivoted_data['interval_end'] = pd.to_datetime(pivoted_data['interval_end'])
    pivoted_data = pivoted_data.sort_values(['user_in_role_id', 'interval_end'])
    return pivoted_data
