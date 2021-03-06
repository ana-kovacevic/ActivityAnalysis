from hmmlearn.hmm import GaussianHMM
import numpy as np
import data_preparation as dp
import pandas as pd


def get_optimal_hmms_for_users_single_variate(data, users, cov_type):
    optimal_hmms_single_variate = {}
    subfactor_activities = dp.get_dict_ges_activities()
    for user in users:
        dict_activity = {}
        for subfactor, activities in subfactor_activities.items():
            for activity in activities:
                prepared_data = dp.prepare_data(data,user,[activity])
                best_value, best_model = optimize_number_of_clusters(prepared_data.iloc[: ,2:], list(range(2,11)), cov_type)
                dict_activity.update({activity: best_model})
        dict_user={user:dict_activity}
        optimal_hmms_single_variate.update(dict_user)
    return optimal_hmms_single_variate


def get_optimal_hmms_for_users_multi_variate(data, users, cov_type):
    optimal_hmms_multi_variate = {}
    subfactor_activities = dp.get_dict_ges_activities()
    for user in users:
        dict_subfactor={}
        for subfactor in subfactor_activities.keys():
                activities=subfactor_activities[subfactor]
                prepared_data = dp.prepare_data(data,user,activities)
                best_value, best_model = optimize_number_of_clusters(prepared_data.iloc[: ,2:], list(range(2,11)), cov_type)
                dict_subfactor.update({subfactor:{'model':best_model, 'activities':activities}})
        dict_user={user:dict_subfactor}
        optimal_hmms_multi_variate.update(dict_user)
    return optimal_hmms_multi_variate


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


def optimize_number_of_clusters(data, range_of_clusters, cov_type):
    '''    
    :param data: prepared data (values of activities by columns) 
    :param range_of_clusters: range of best number expected e.g. 2:10
    :return:
     Optimizes number of clusters for single citizen
     This is helper method for get_optimal_hmms methods (they work for more citizens)
    '''
    best_value=np.inf # create for maximization and minimization
    best_model=None
    #pivoted_data = prepare_data(data, user, [ac]) old version for data preparation
    for n_states in range_of_clusters:
        model = GaussianHMM(n_components=n_states, covariance_type=cov_type, n_iter=1000).fit(data)
        log_likelihood = model.score(data)
        criteria=aic_criteria(data, log_likelihood, model)
        if criteria < best_value:
            best_value, best_model = criteria, model
    return best_value, best_model

### TODO add method for research purposes
#def log_hmm_optimization():


def log_optimal_hmms_for_users_single_variate(data, users, cov_type):
    optimal_hmms_single_variate = {}
    subfactor_activities = dp.get_dict_ges_activities()
    for user in users:
        dict_activity = {}
        for subfactor, activities in subfactor_activities.items():
            for activity in activities:
                prepared_data = dp.prepare_data(data,user,[activity])
                log = optimize_number_of_clusters(prepared_data.iloc[: ,2:], list(range(2,11)), cov_type)



    return log


def log_optimal_hmms_for_users_multi_variate(data, users, cov_type):
    optimal_hmms_multi_variate = {}
    subfactor_activities = dp.get_dict_ges_activities()
    for user in users:
        dict_subfactor={}
        for subfactor in subfactor_activities.keys():
                activities=subfactor_activities[subfactor]
                prepared_data = dp.prepare_data(data,user,activities)
                best_value, best_model = optimize_number_of_clusters(prepared_data.iloc[: ,2:], list(range(2,11)), cov_type)
                dict_subfactor.update({subfactor:{'model':best_model, 'activities':activities}})
        dict_user={user:dict_subfactor}
        optimal_hmms_multi_variate.update(dict_user)
    return optimal_hmms_multi_variate


def log_activity_results(data, users, range_of_clusters, cov_type, single_multi):
    '''
    :param data: prepared data (values of activities by columns)
    :param range_of_clusters: range of best number expected e.g. 2:10
    :return:
     Optimizes number of clusters for single citizen
     This is helper method for get_optimal_hmms methods (they work for more citizens)
    '''
    import pickle
    log_results = []
    subfactor_activities = dp.get_dict_ges_activities()
    for user in users:
        for subfactor, activities in subfactor_activities.items():
            for activity in activities:
                prepared_data = dp.prepare_data(data, user, [activity])
                #log = optimize_number_of_clusters(prepared_data.iloc[:, 2:], list(range(2, 11)), cov_type)
                # pivoted_data = prepare_data(data, user, [ac]) old version for data preparation
                for n_states in range_of_clusters:
                    model = GaussianHMM(n_components=n_states, covariance_type=cov_type, n_iter=1000).fit(prepared_data.iloc[: ,1:])
                    log_likelihood = model.score(data)
                    criteria_bic = bic_criteria(data, log_likelihood, model)
                    criteria_aic = aic_criteria(data, log_likelihood, model)
                    aic_bic_dict = {'user': user, 'activity': activity, 'n_states': n_states, 'BIC': criteria_bic,
                                    'AIC': criteria_aic}
                    log_results.append(aic_bic_dict)
                    if single_multi == 'single':
                        path = 'Experimental_Evaluation/Models/user_' + user + 'activity_' + activity + '_n_states_' + n_states + '.pkl'
                    if single_multi == 'multi':
                        path = 'Experimental_Evaluation/Models/user_' + user + 'sub_factor_' + activity + '_n_states_' + n_states + '.pkl'
                    pickle.dump(model, path)

    if single_multi == 'single':
        log_path = 'Experimental_Evaluation/single_variate_log.csv'
    if single_multi == 'multi':
        log_path = 'Experimental_Evaluation/multi_variate_log.csv'

    log = pd.DataFrame(log_results)
    log.to_csv(log_results, log_path)
    return log


def bic_criteria(data, log_likelihood, model):
    '''
    :param data: 
    :param log_likelihood: 
    :param model: 
    :return: 
    '''
    n_features = data.shape[1]  ### here adapt for multi-variate
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
    n_features = data.shape[1]  ### here adapt for multi-variate
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
