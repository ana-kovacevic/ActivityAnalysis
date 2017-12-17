from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, DayLocator
import numpy as np
import datetime as dt
import pandas as pd
import json
import Models_Score as ms


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
    plt.rcParams["figure.figsize"]=[20.0, 10.0]
    plt.savefig( 'Plots/single_variate/''citizen_id_' + str(user)+ '_activity_'+ activity +'.png')


########################### Plot single series with colored clusters

def create_oneSeries_single_variate_plot(data, user, activity):

    plt.clf()
    data = data.loc[(data['user_in_role_id'] == user) & (data['detection_variable_name'].isin([activity]))]
    num_clusters = len(data.cluster.unique())
    dates = data['interval_end']
    hidden_states = data['cluster']
    a = hidden_states
    x = dates
    Y = data['measure_value']

    colours = cm.rainbow(np.linspace(0, 1, num_clusters))

    for a, x1, x2, y1, y2 in zip(a[1:], x[:-1], x[1:], Y[:-1], Y[1:]):
        plt.plot_date([x1, x2], [y1, y2], ".-", c=colours[a])
        plt.grid(True)

    plt.subplots_adjust(top=0.89, left=0.1, right=0.9, bottom=0.12)
    plt.suptitle("User_in_role_id: " + str(user) + "     Activity: " + activity)
    plt.rcParams["figure.figsize"] = [20.0, 10.0]
    plt.savefig('Plots/transitions/''Transition_citizen_id_' + str(user)+ '_activity_'+ activity +'.png')


    plt.show()




#### Read clustered data
clustered_data = pd.read_csv('Data/clustered_data/single_variate_clusters.csv')


#### Get unique users and theirs activities
#users = clusterd_data.user_in_role_id.unique()
json_users_activities_models=open('Models/HMM/JSON/single_variate_hmms.json').read()
users_activities_models=json.loads(json_users_activities_models)
users_activities = ms.get_users_activities(users_activities_models)

#### generate single varite plots where each state is on another subplot
for user, activities in users_activities.items():
    for activity in activities:
        create_single_variate_plot(clustered_data, user, activity)

### generate single variate plots where Time Series is colored by states
for user, activities in users_activities.items():
    for activity in activities:
        create_oneSeries_single_variate_plot(clustered_data, user, activity)

'''
################################################
data = clustered_data.loc[(clustered_data['user_in_role_id'] == 66) & (clustered_data['detection_variable_name'].isin(['physicalactivity_calories']))]
num_clusters = len(data.cluster.unique())
dates = data['interval_end']
hidden_states = data['cluster']
create_oneSeries_single_variate_plot(data, 66, 'physicalactivity_calories')
'''
