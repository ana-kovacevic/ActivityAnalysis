from hmmlearn.hmm import GaussianHMM
import pandas as pd
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, DayLocator
import numpy as np
import datetime as dt

data = pd.read_csv('activities_out.csv')

def get_users_activities(data, user):
    user_data=data[data['user_in_role_id']==user]
    d = user_data.groupby(['user_in_role_id', 'detection_variable_name'])['measure_value'].count()
    d.rename(columns={'measure_value':'count_measure_value'}, inplace=True)
    d=pd.DataFrame(d)
    return d

#### Check all activities and counts of appearences for single user
def select_pivot_users_activities(data, user, activities):
    user_data=data[data['user_in_role_id']==user]
    user_data=user_data[user_data['detection_variable_name'].isin(activities)]
    pivot_data = user_data.pivot_table(index=['user_in_role_id', 'interval_end'], columns='detection_variable_name',values='Normalised')
    return pivot_data


#### Create single user single multiple activities cluster

activities=['sleep_awake_time','sleep_deep_time', 'sleep_light_time', 'sleep_tosleep_time']
user=66
pivoted_data=select_pivot_users_activities(data, user, activities)

pivoted_data = pivoted_data.reset_index()
pivoted_data['interval_end']=pd.to_datetime(pivoted_data['interval_end'])

#pivoted_data
#pivoted_data = pivoted_data['interval_end'].dt.date

pivoted_data = pivoted_data.sort_values(['user_in_role_id','interval_end'])
#pivoted_data.head()
#pivoted_data[['sleep_awake_time','sleep_deep_time', 'sleep_light_time', 'sleep_tosleep_time']]

#pivoted_data=pivoted_data.iloc[:,2:]

model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=1000).fit(pivoted_data.iloc[:,2:])
hidden_states=model.predict(pivoted_data.iloc[:,2:])

'''
Y=pivoted_data['sleep_awake_time']
Z=pivoted_data['sleep_deep_time']
W=pivoted_data['sleep_light_time']
X=pivoted_data['sleep_tosleep_time']
'''
dates=(pivoted_data['interval_end']).dt.to_pydatetime()





#### Print model parameters
#maybe add whaole covariances
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

##### Plot model results (means and variances)


### Subplot the states multi-variate single user - By States
fig, axs = plt.subplots(model.n_components, sharex=True, sharey=True)
#colours = cm.rainbow(np.linspace(0, 1, model.n_components))
colours = cm.rainbow(np.linspace(0, 1, len(activities)))
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
axs.flatten()[-1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=2)
plt.show()



############## Check but hard to work for multivariate
############################### Color multivariate time series by States

a = hidden_states
x = dates
y = Y
for a, x1, x2, y1, y2 in zip(a[1:], x[:-1], x[1:], y[:-1], y[1:]):
    if a == 0:
        plt.plot([x1, x2], [y1, y2], 'r')
    elif a == 1:
        plt.plot([x1, x2], [y1, y2], 'g')
    else:
        plt.plot([x1, x2], [y1, y2], 'b')

a = hidden_states
x = dates
y = Z
for a, x1, x2, y1, y2 in zip(a[1:], x[:-1], x[1:], y[:-1], y[1:]):
    if a == 0:
        plt.plot([x1, x2], [y1, y2], 'r')
    elif a == 1:
        plt.plot([x1, x2], [y1, y2], 'g')
    else:
        plt.plot([x1, x2], [y1, y2], 'b')

plt.show()



#### Create single user single activity cluster
results_single_hmm =[]
for user in users:
    user_data=data[data['user_in_role_id']==user]
    activities=user_data['detection_variable_name'].unique()
    for ac in activities:
        user_ac=user_data[user_data['detection_variable_name']==ac]
        if user_ac.shape[0]>30:
            normalized=user_ac[['Normalised']]
            original=user_ac[['measure_value']]
            dates=user_ac[['interval_end']]
            model = GaussianHMM(n_components=5, covariance_type="diag", n_iter=1000).fit(normalized)
            predictions=model.predict(normalized)
            results_single_hmm.append((user, ac, model, predictions, dates, normalized, original))




### Find optimal number of clusters

activities=results_single_hmm[4][5]
ks=range(2,11)

#find_optimal_HMM(activities,ks)
#draw_clusters(results_single_hmm[4])


# selektuj uset/activity
def find_optimal_HMM(activities, cluster_range):
    scores=[]
    for k in cluster_range:
        model=GaussianHMM(n_components=k, covariance_type="diag", n_iter=1000).fit(activities)
        log_likelihood=model.score(activities)
        bic=2*log_likelihood-k*np.log(len(activities))
        scores.append((k, log_likelihood, bic))
    return scores






model.score()


### Print or save clusters
#for result in results_single_hmm:
#    draw_clusters(result)

### Ploting complete behavior


#### Subplot of states sigle variate single user

def draw_clusters(results, path_store='Single_Activity_Clusters/De-normalized/'):
    model=results[2]
    hidden_states=results[3]
    dates=results[4]
    Y=results[6]
    fig, axs = plt.subplots(model.n_components, sharex=True, sharey=True)
    colours = cm.rainbow(np.linspace(0, 1, model.n_components))
    for i, (ax, colour) in enumerate(zip(axs, colours)):
        # Use fancy indexing to plot data in each state.
        mask = hidden_states == i
        ax.plot_date(dates[mask], Y[mask], ".-", c=colour)
        ax.set_title("{0}th hidden state".format(i+1))
        # Format the ticks.
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.xaxis.set_minor_locator(DayLocator())
        ax.grid(True)
    plt.suptitle("User_in_role_id: " + str(results[0]) + "     Activity: "+str(results[1]))
    #plt.savefig(path_store + 'user_' + str(results[0])+ '_activity_'+str(results[1])+'.png', bbox_inches='tight')
    #plt.show()


'''






oneexample = data[data['user_in_role_id'] ==  68] # get one user
oneexample = oneexample[oneexample['detection_variable_name'] =='walk_distance'] # select one activity

oneexample
Y = oneexample[['Normalised']]
X = oneexample[['measure_value']]

Dates=oneexample[['interval_end']]

type(['interval_end'])

###################################### Make an HMM instance and execute fit

#################### Train on Normalised value (Y)
model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000).fit(Y)

# Predict the optimal sequence of internal hidden state
hidden_states = model.predict(Y)

print("done")

print("Transition matrix")
print(model.transmat_)
print()

#### Print model parameters

print("Transition matrix")
print(model.transmat_)
print()

print("Means and vars of each hidden state")
for i in range(model.n_components):
    print("{0}th hidden state".format(i))
    print("mean = ", model.means_[i])
    print("var = ", np.diag(model.covars_[i]))
    print()

dates=oneexample['interval_end']
dates = pd.to_datetime(dates)

#### Subplot of states
fig, axs = plt.subplots(model.n_components, sharex=True, sharey=True)
colours = cm.rainbow(np.linspace(0, 1, model.n_components))
for i, (ax, colour) in enumerate(zip(axs, colours)):
    # Use fancy indexing to plot data in each state.
    mask = hidden_states == i
    ax.plot_date(dates[mask], Y[mask], ".-", c=colour)
    ax.set_title("{0}th hidden state".format(i))

    # Format the ticks.
    #ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_minor_locator(MonthLocator())
    ax.xaxis.set_minor_locator(DayLocator())

    ax.grid(True)

plt.show()
############################### Color time series by States
tsY = pd.DataFrame(Y['Normalised'])
tsY['Date'] = dates
selY = tsY.ix[:].dropna()

selY['Date'] = pd.to_datetime(selY['Date'])
selY.set_index('Date', inplace=True)
stateY = (pd.DataFrame(hidden_states, columns=['state'], index=selY.index)
          .join(selY, how='inner')
          .assign(vol_cret=selY.Normalised.cumsum())
          .reset_index(drop=False)
          .rename(columns={'index':'Date'}))
print(stateY.head())

a = stateY['state']
x = stateY.index
y = stateY['Normalised']
for a, x1, x2, y1, y2 in zip(a[1:], x[:-1], x[1:], y[:-1], y[1:]):
    if a == 0:
        plt.plot([x1, x2], [y1, y2], 'r')
    elif a == 1:
        plt.plot([x1, x2], [y1, y2], 'g')
    else:
        plt.plot([x1, x2], [y1, y2], 'b')

plt.show()






'''