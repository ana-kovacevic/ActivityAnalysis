from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, DayLocator
import numpy as np
import datetime as dt
import pandas as pd
import json



######### This method is defined in Models_Score and shoud be removed
######### Instead of re-defining, it should be imported and called from here
def get_users_activities(users_activities_models):
    '''
    :param users_activities_models: 
    :return: 
    '''
    dict={}
    for user, activities_models in users_activities_models.items():
        activity_list=[]
        for activity in activities_models.keys():
            activity_list.append(activity)
        dict.update({int(user):activity_list})
    return dict



#type(clustered_data['interval_end'][0])

##### Create plot

def create_single_variate_plot(data, user, activity):
    data=data.loc[(data['user_in_role_id'] == user) & (data['detection_variable_name'].isin([activity]))]
    num_clusters = len(data.cluster.unique())
    hidden_states = data['cluster']
    dates=pd.to_datetime(data['interval_end'])
    fig, axs = plt.subplots(num_clusters, sharex=True, sharey=True)
    colours = cm.rainbow(np.linspace(0, 1, num_clusters))
    i=0
    lines=[]
    for ax in axs:
        mask = hidden_states == i
        Y = data['measure_value']
        ax.plot_date(dates[mask], Y[mask], ".-", c=colours[i], label =activity)
        i=i+1
        ax.set_title("{0}. Behaviour".format(i))
        # Format the ticks.
        # ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.xaxis.set_minor_locator(DayLocator())
        ax.grid(True)
        # plt.suptitle("User_in_role_id: " + str(results[0]) + "     Activity: "+str(results[1]))
        # plt.savefig(path_store + 'user_' + str(results[0])+ '_activity_'+str(results[1])+'.png', bbox_inches='tight')
    fig.subplots_adjust(top=0.89, left=0.1, right=0.9, bottom=0.12, hspace = 0.25)
    #axs.flatten()[-1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=2)
    plt.suptitle("User_in_role_id: " + str(user) + "     Activity: "+ activity)
    plt.rcParams["figure.figsize"]=[10.0, 10.0]
    plt.savefig( 'Plots/single_variate/''citizen_id_' + str(user)+ '_activity_'+ activity +'.png')
    


#### Read clustered data
clustered_data = pd.read_csv('Data/clustered_data/single_variate_clusters.csv')
#clusterd_data.head()

#### Get unique users and theirs activities
#users = clusterd_data.user_in_role_id.unique()
json_users_activities_models=open('Models/HMM/JSON/single_variate_hmms.json').read()
users_activities_models=json.loads(json_users_activities_models)
users_activities = get_users_activities(users_activities_models)

users_activities


for user, activities in users_activities.items():
    for activity in activities:
        create_single_variate_plot(clustered_data, user, activity)


'''
user66 = clusterd_data.loc[(clusterd_data['user_in_role_id'] == 66) & clusterd_data['detection_variable_name'].isin(['physicalactivity_calories'])]
num_clusters = len(user66.cluster.unique())
hidden_states = user66['cluster']
dates = user66['interval_end']
create_single_variate_plot(clustered_data, 66, 'sleep_deep_time')
'''
