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
    plt.rcParams["figure.figsize"]=[10.0, 10.0]
    plt.savefig( 'Plots/single_variate/''citizen_id_' + str(user)+ '_activity_'+ activity +'.png')



#### Read clustered data
clustered_data = pd.read_csv('Data/clustered_data/single_variate_clusters.csv')
#clusterd_data.head()

#### Get unique users and theirs activities
#users = clusterd_data.user_in_role_id.unique()
json_users_activities_models=open('Models/HMM/JSON/single_variate_hmms.json').read()
users_activities_models=json.loads(json_users_activities_models)
users_activities = ms.get_users_activities(users_activities_models)


for user, activities in users_activities.items():
    for activity in activities:
        create_single_variate_plot(clustered_data, user, activity)



########################### Plot single series with colored clusters

'''
user66 = clustered_data.loc[(clustered_data['user_in_role_id'] == 66) & clustered_data['detection_variable_name'].isin(['physicalactivity_calories'])]
num_clusters = len(user66.cluster.unique())
hidden_states = user66['cluster']
dates = user66['interval_end']
values = user66['measure_value']



a = hidden_states
x = dates
y = values
colors=cm.rainbow(np.linspace(0,1,num_clusters))

for a, x1, x2, y1, y2 in zip(a[1:], x[:-1], x[1:], y[:-1], y[1:]):
    for h in hidden_states:
        
        if a == 0:
            plt.plot([x1, x2], [y1, y2], 'r')
        elif a ==1 :
            plt.plot([x1, x2], [y1, y2], 'g')
        else:
            plt.plot([x1, x2], [y1, y2], 'b')

plt.show()

'''




