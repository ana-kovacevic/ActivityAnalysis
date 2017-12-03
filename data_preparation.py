import pandas as pd


### Maybe add this to data exploration
def get_users_activities(data, user):
    '''
    :param data: data 
    :param user: user elderly
    :return: all activities
    
    Gets all activities for specified user and returns activity names and counts
    Exploratory method for selection of users/activities for modelling
    '''
    user_data=data[data['user_in_role_id']==user]
    d = user_data.groupby(['user_in_role_id', 'detection_variable_name'])['measure_value'].count()
    #d.rename(columns={'measure_value':'count_measure_value'}, inplace=True)
    d=pd.DataFrame(d)
    return d

def select_pivot_users_activities(data, user, activities):
    '''
    Pivots multivariate data - each activity becomes column
    Unnecessary step for single variate time series - maybe remove and adjust prepare data method
    '''
    user_data=data[data['user_in_role_id']==user]
    user_data=user_data[user_data['detection_variable_name'].isin(activities)]
    pivot_data = user_data.pivot_table(index=['user_in_role_id', 'interval_end'], columns='detection_variable_name',values='Normalised')
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

'''
##########################################################################
DATA PREPARATION MULTIVARIATE
###########################################################################
'''
#### Check all activities and counts of appearences for single user
def select_pivot_users_activities_multi_variate(data, user, activities, activity_extremization):
    user_data=data[data['user_in_role_id']==user]
    user_data=user_data[user_data['detection_variable_name'].isin(activities)]
    pivot_data = user_data.pivot_table(index=['user_in_role_id', 'interval_end'], columns='detection_variable_name',values='Normalised')
    invert_mins(activity_extremization, pivot_data) # difference with single variate - all activities are set to max extremization
    return pivot_data


def create_multi_variate_clusters(data, user, activities, activity_extremization):
    pivoted_data = select_pivot_users_activities_multi_variate(data, user, activities, activity_extremization)
    pivoted_data = pivoted_data.reset_index()
    pivoted_data['interval_end'] = pd.to_datetime(pivoted_data['interval_end'])
    pivoted_data = pivoted_data.sort_values(['user_in_role_id', 'interval_end'])
    model = GaussianHMM(n_components=5, covariance_type="diag", n_iter=1000).fit(pivoted_data.iloc[:, 2:])
    hidden_states = model.predict(pivoted_data.iloc[:, 2:])
    return(model, pivoted_data, activities, hidden_states)


def invert_mins(activity_extremization, pivoted_data):
    keys=list(activity_extremization.keys())
    for key in keys:
        if activity_extremization[key]=='min':
           pivoted_data[key]=pivoted_data[key].apply(lambda x: 1-x)
