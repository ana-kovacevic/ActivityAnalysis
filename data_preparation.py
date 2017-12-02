import pandas as pd
def get_users_activities(data, user):
    '''
    :param data: data 
    :param user: user elderly
    :return: all activities
    gets all activities for specified user and returns activity names and counts
    '''
    user_data=data[data['user_in_role_id']==user]
    d = user_data.groupby(['user_in_role_id', 'detection_variable_name'])['measure_value'].count()
    #d.rename(columns={'measure_value':'count_measure_value'}, inplace=True)
    d=pd.DataFrame(d)
    return d

