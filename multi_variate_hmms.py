from hmmlearn.hmm import GaussianHMM
import pandas as pd


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


'''
###########################################################################
PROGRAM LOGIC
###########################################################################
'''

data = pd.read_csv('activities_out.csv')

user=68
activities=['sleep_awake_time','sleep_deep_time', 'sleep_light_time', 'sleep_tosleep_time', 'sleep_wakeup_num']
activity_extremization = {'sleep_light_time':'max', 'sleep_deep_time':'max', 'sleep_awake_time':'min', 'sleep_wakeup_num':'min', 'sleep_tosleep_time':'min'}
activity_weights = {'sleep_light_time':0.1, 'sleep_deep_time':0.3, 'sleep_awake_time':0.1, 'sleep_wakeup_num':0.3, 'sleep_tosleep_time':0.2}

### Create and plot clusters
model, pivoted_data, activities, hidden_states=create_multi_variate_clusters(data, user, activities, activity_extremization)
plot_multivariate_clusters(model, pivoted_data, activities, hidden_states)


print_hmm_params(model)


### Assign grades to clusters (weighted sum of weights and activity means for each cluster)

grades=calculate_grades(model=model, clusters=hidden_states, activity_weights=activity_weights)


''' 
Group and Plot factor on monthly level
'''

dates=pivoted_data['interval_end']
#dat=pd.Series(dates, index=list(range(len(dates))), name='dates')
fac=pd.Series(grades, index=list(range(len(grades))), name='grades')

dates_grades=pd.concat([dates, fac], axis=1)

dates_grades['year'] = [y.year for y in dates_grades['interval_end']]
dates_grades['month'] = [m.month for m in dates_grades['interval_end']]
dates_grades['day'] = [d.day for d in dates_grades['interval_end']]


factor=pd.groupby(dates_grades, by=[dates_grades.month, dates_grades.year]).agg('mean')

print(factor)

plot_single_series('motility', factor, dates)

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










##### For single user and each activity -
# 5 clusters,











# assign grades,
#
# assign interactivity weights

# calculate higher level factors


#### BIC calculation ###

#!!!!!!!!!!!!!!!!!!!!!!! CHECK THIS !!!!!!!!!!!!!!!
'''
FOR HMM

Nparams = size(model.A,2)*(size(model.A,2)-1) +
          size(model.pi,2)-1) +
          size(model.emission.T,1)*(size(model.emission.T,2)-1)
Nparams = 13
BIC = -2*logLike + num_of_params*log(length(x))
Nparams = Num_of_states*(Num_of_States-1) - Nbzeros_in_transition_matrix
'''
'''
#FOR GMM
nParam = (k_mixtures – 1) + (k_mixtures * NDimensions ) + k_mixtures * Ndimensions  %for daigonal covariance matrices
nParam = (k_mixtures – 1) + (k_mixtures * NDimensions ) + k_mixtures * NDimensions * (NDimensions+1)/2; %for full covariance matrices
'''


#### ARIMA #####

from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(series, order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())







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


'''
