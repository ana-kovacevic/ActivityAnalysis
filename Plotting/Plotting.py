from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, DayLocator
import numpy as np
import datetime as dt
import pandas as pd
import json




'''
##########################################################################
VIZUALIZATION
###########################################################################
'''
### plots clusters for single activity (one plot)
def plot_single_variate_cluster(activity, model, hidden_states, values, dates):
    ### Subplot the states multi-variate single user - By States gyf
    fig, axs = plt.subplots(model.n_components, sharex=True, sharey=True)
    colours = cm.rainbow(np.linspace(0, 1, model.n_components))
    #colours = cm.rainbow(np.linspace(0, 1, len(activities)))
    #dates=pivoted_data['interval_end']
    i=0
    lines=[]
    for ax in axs:
        # Use fancy indexing to plot data in each state.
        mask = hidden_states == i
        Y = values
        ax.plot_date(dates[mask], Y[mask], ".-", c=colours[i], label =activity)
        i=i+1
        ax.set_title("{0}th hidden state".format(i))
        # Format the ticks.
        #ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.xaxis.set_minor_locator(DayLocator())
        ax.grid(True)
        #plt.suptitle("User_in_role_id: " + str(results[0]) + "     Activity: "+str(results[1]))
        #plt.savefig(path_store + 'user_' + str(results[0])+ '_activity_'+str(results[1])+'.png', bbox_inches='tight')
    fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)
    axs.flatten()[-1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=2)
    plt.show()

def plot_single_series(activity, values, dates):
    ### Subplot the states multi-variate single user - By States
    #fig, axs = plt.subplots(model.n_components, sharex=True, sharey=True)
    colours = cm.rainbow(np.linspace(0, 1, 1))
    #colours = cm.rainbow(np.linspace(0, 1, len(activities)))
    #dates=pivoted_data['interval_end']
    #i=0
    #lines=[]
    #for ax in axs:
        # Use fancy indexing to plot data in each state.
     #   mask = hidden_states == i
    Y = values
    ax=plt.plot_date(dates, Y, ".-", c=colours[0], label =activity )
    ax.xaxis.set_minor_locator(MonthLocator())
    ax.xaxis.set_minor_locator(DayLocator())
    ax.grid(True)
        #plt.suptitle("User_in_role_id: " + str(results[0]) + "     Activity: "+str(results[1]))
        #plt.savefig(path_store + 'user_' + str(results[0])+ '_activity_'+str(results[1])+'.png', bbox_inches='tight')
    #fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)
    #axs.flatten()[-1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=2)
    plt.show()


### plots clusters for multiple activities (multiple plots)
### Reminder - add writing to disk
def plot_single_variate_multiple_activities(res):
    for ac in res.keys():
        activity=res[ac]['name']
        model = res[ac]['model']
        hidden_states = res[ac]['clusters']
        values=res[ac]['values']
        dates = res[ac]['dates']
        plot_single_variate_cluster(activity, model, hidden_states, values, dates)


def print_hmm_params(model):
    print("Transition matrix")
    print(model.transmat_)
    print()

    print("Means and vars of each hidden state")
    for i in range(model.n_components):
        print("{0}th hidden state".format(i))
        print("mean = ", model.means_[i])
        print("var = ", np.diag(model.covars_[i]))
        print()

    print_hmm_params(model)

'''
##########################################################################
MULTI VARIATE VIZUALIZATION
###########################################################################

'''

def plot_multivariate_clusters(model, pivoted_data, activities, hidden_states):
    ### Subplot the states multi-variate single user - By States
    fig, axs = plt.subplots(model.n_components, sharex=True, sharey=True)
    #colours = cm.rainbow(np.linspace(0, 1, model.n_components))
    colours = cm.rainbow(np.linspace(0, 1, len(activities)))
    dates=pivoted_data['interval_end']
    i=0
    lines=[]
    for ax in axs:
        # Use fancy indexing to plot data in each state.
        mask = hidden_states == i
        i=i+1
        for j in range(len(activities)):
            Y = pivoted_data[activities[j]]
            lines.append( ax.plot_date(dates[mask], Y[mask], ".-", c=colours[j], label =activities[j]))
        ax.set_title("{0}th hidden state".format(i))
        # Format the ticks.
        #ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.xaxis.set_minor_locator(DayLocator())
        ax.grid(True)
        #plt.suptitle("User_in_role_id: " + str(results[0]) + "     Activity: "+str(results[1]))
        #plt.savefig(path_store + 'user_' + str(results[0])+ '_activity_'+str(results[1])+'.png', bbox_inches='tight')
    fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)
    axs.flatten()[-1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.9), ncol=2)
    plt.show()

    def plot_single_series(activity, values, dates):
        ### Subplot the states multi-variate single user - By States
        # fig, axs = plt.subplots(model.n_components, sharex=True, sharey=True)
        colours = cm.rainbow(np.linspace(0, 1, 1))
        # colours = cm.rainbow(np.linspace(0, 1, len(activities)))
        # dates=pivoted_data['interval_end']
        # i=0
        # lines=[]
        # for ax in axs:
        # Use fancy indexing to plot data in each state.
        #   mask = hidden_states == i
        Y = values
        ax = plt.plot_date(dates, Y, ".-", c=colours[0], label=activity)
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.xaxis.set_minor_locator(DayLocator())
        ax.grid(True)
        # plt.suptitle("User_in_role_id: " + str(results[0]) + "     Activity: "+str(results[1]))
        # plt.savefig(path_store + 'user_' + str(results[0])+ '_activity_'+str(results[1])+'.png', bbox_inches='tight')
        # fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)
        # axs.flatten()[-1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=2)
        plt.show()

    def print_hmm_params(model):
        print("Transition matrix")
        print(model.transmat_)
        print()

        print("Means and vars of each hidden state")
        for i in range(model.n_components):
            print("{0}th hidden state".format(i))
            print("mean = ", model.means_[i])
            print("var = ", np.diag(model.covars_[i]))
            print()