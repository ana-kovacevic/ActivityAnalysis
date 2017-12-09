from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, DayLocator
import numpy as np
import datetime as dt
import pandas as pd
import json

#### Read clustered data
clusterd_data = pd.read_csv('Data/clustered_data/single_variate_clusters.csv')
clusterd_data.head()

#### Get unique users and theirs activities
users = clusterd_data.user_in_role_id.unique()
json_users_activities_models=open('Models/HMM/JSON/single_variate_hmms.json').read()
users_activities_models=json.loads(json_users_activities_models)

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
users_activities = get_users_activities(users_activities_models)

###### Hard-coded values for testing plot
###### This should be generalised for all users and their activities
user66 = clusterd_data.loc[(clusterd_data['user_in_role_id'] == 66) & clusterd_data['detection_variable_name'].isin(['physicalactivity_calories'])]
num_clusters = len(user66.cluster.unique())
hidden_states = user66['cluster']
dates = user66['interval_end']

##### Create plot

fig, axs = plt.subplots(num_clusters, sharex=True, sharey=True)
colours = cm.rainbow(np.linspace(0, 1, num_clusters))

i=0
lines=[]
for ax in axs:

    mask = hidden_states == i
    Y = user66['measure_value']
    ax.plot_date(dates[mask], Y[mask], ".-", c=colours[i], label ='physicalactivity_calories')
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
axs.flatten()[-1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=2)
plt.suptitle("User_in_role_id: " + str(66) + "     Activity: "+ 'physicalactivity_calories')
plt.rcParams["figure.figsize"]=[10.0, 10.0]
plt.savefig( 'user_' + str(66)+ '_activity_'+ 'physicalactivity_calories' +'.png', )
plt.show()
