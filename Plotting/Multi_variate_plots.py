from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, DayLocator
import numpy as np
import datetime as dt
import pandas as pd
import json
import Models_Score as ms



def create_multi_variate_plot(data, user, ges, activities):
    ### Subplot the states multi-variate single user - By States
    num_clusters = len(data.cluster.unique())
    hidden_states = data['cluster']
    dates = data['interval_end']

    fig, axs = plt.subplots(num_clusters, sharex=True, sharey=True)
    #colours = cm.rainbow(np.linspace(0, 1, model.n_components))
    colours = cm.rainbow(np.linspace(0, 1, len(activities)))
    # dates=pivoted_data['interval_end']
    i=0
    lines=[]
    for ax in axs:
        # Use fancy indexing to plot data in each state.
        mask = hidden_states == i
        i=i+1
        for j in range(len(activities)):
            Y = data[activities[j]]
            lines.append( ax.plot_date(dates[mask], Y[mask], ".-", c=colours[j], label =activities[j]))

        ax.set_title("{0}. Behaviour".format(i))
        # Format the ticks.
        #ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.xaxis.set_minor_locator(DayLocator())
        ax.grid(True)

    fig.subplots_adjust(top=0.85, left=0.1, right=0.9, bottom=0.05, hspace = 0.25)
    plt.suptitle("User_in_role_id: " + str(user) + "\nSubfactor: "+ ges, x= 0.1, horizontalalignment = 'left')
    axs.flatten()[1].legend(loc='upper right', bbox_to_anchor=(1, 2.89), ncol=2)
    plt.rcParams["figure.figsize"]=[13.0, 10.0]
    plt.savefig( 'Plots/multi_variate/''citizen_id_' + str(user)+ '_' + ges +'.png')
   # plt.show()


#### CUMSUM

def create_multi_variate_cumulative_plot(data, user, ges, activities):
    num_clusters = len(data.cluster.unique())
    hidden_states = data['cluster']
    dates = data['interval_end']


    fig, axs = plt.subplots(num_clusters, sharex=True, sharey=True)
    #colours = cm.rainbow(np.linspace(0, 1, model.n_components))
    colours = cm.rainbow(np.linspace(0, 1, len(activities)))
    # dates=pivoted_data['interval_end']
    i=0
    lines=[]
    for ax in axs:
        # Use fancy indexing to plot data in each state.
        mask = hidden_states == i
        i=i+1
        for j in range(len(activities)):
            Y = data[activities[j]].cumsum()
            lines.append( ax.plot_date(dates[mask], Y[mask], ".-", c=colours[j], label =activities[j]))

        ax.set_title("{0}. Behaviour".format(i))
        # Format the ticks.
        #ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.xaxis.set_minor_locator(DayLocator())
        ax.grid(True)

    fig.subplots_adjust(top=0.85, left=0.1, right=0.9, bottom=0.05, hspace = 0.25)
    plt.suptitle("User_in_role_id: " + str(user) + "\nSubfactor: "+ ges, x= 0.1, horizontalalignment = 'left')
    axs.flatten()[1].legend(loc='upper right', bbox_to_anchor=(1, 2.89), ncol=2)
    plt.rcParams["figure.figsize"]=[13.0, 10.0]
    plt.savefig( 'Plots/multi_variate_cumulative/''cum_citizen_id_' + str(user)+ '_' + ges +'.png')
    plt.show()

### read json
json_users_activities_models = open('Models/HMM/JSON/multi_variate_hmms.json').read()
users_activities_models = json.loads(json_users_activities_models)
users_ges_activities = ms.get_users_ges_activities(users_activities_models)


for user, ges_activities in users_ges_activities.items():
    for ges, activities in ges_activities.items():
        path = 'Data/clustered_data/multi_variate_clusters/citizen_id_' + str(user) + '_' + ges + '.csv'
        data = pd.read_csv(path)
        create_multi_variate_plot(data, user, ges, activities)


for user, ges_activities in users_ges_activities.items():
    for ges, activities in ges_activities.items():
        path = 'Data/clustered_data/multi_variate_clusters/citizen_id_' + str(user) + '_' + ges + '.csv'
        data = pd.read_csv(path)
        create_multi_variate_cumulative_plot(data, user, ges, activities)


'''
user66 = pd.read_csv('Data/clustered_data/multi_variate_clusters/citizen_id_66_physical_activity.csv')
num_clusters = len(user66.cluster.unique())
hidden_states = user66['cluster']
dates = user66['interval_end']
activities = ['physicalactivity_calories','physicalactivity_intense_time', 'physicalactivity_moderate_time', 'physicalactivity_soft_time']
'''

