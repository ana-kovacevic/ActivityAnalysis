from hmmlearn.hmm import GaussianHMM
import pandas as pd
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, DayLocator
import numpy as np
import datetime as dt


# gets all activities for specified user and returns activity names and counts
def get_users_activities(data, user):
    user_data=data[data['user_in_role_id']==user]
    d = user_data.groupby(['user_in_role_id', 'detection_variable_name'])['measure_value'].count()
    d.rename(columns={'measure_value':'count_measure_value'}, inplace=True)
    d=pd.DataFrame(d)
    return d

### Pivots multivariate data - each activity becomes column
### Unnecessary step for single variate time series - maybe remove and adjust prepare data method
def select_pivot_users_activities(data, user, activities):
    user_data=data[data['user_in_role_id']==user]
    user_data=user_data[user_data['detection_variable_name'].isin(activities)]
    pivot_data = user_data.pivot_table(index=['user_in_role_id', 'interval_end'], columns='detection_variable_name',values='Normalised')
    return pivot_data


### Takes pivoted data and transforms it in regular DataFrame. Converts dates to date format (for plotting) , sorts data based on dates in order to preserve temporal order
def prepare_data(data, user, ac):
    pivoted_data = select_pivot_users_activities(data, user, [ac])
    pivoted_data = pivoted_data.reset_index()
    pivoted_data['interval_end'] = pd.to_datetime(pivoted_data['interval_end'])
    pivoted_data = pivoted_data.sort_values(['user_in_role_id', 'interval_end'])
    return pivoted_data

### Creates single variate HMM models for each activity.
### This method should only be called. It calls preparation methods and returns dictionary with elements for plotting and mapping to states
def create_single_variate_clusters(data, activities, activity_extremization, activity_weights):
    clusters_activities = {}
    for ac in activities:
        pivoted_data=prepare_data(data, user, ac)
        model = GaussianHMM(n_components=5, covariance_type="full", n_iter=1000).fit(pivoted_data.iloc[:, 2:])
        hidden_states = model.predict(pivoted_data.iloc[:, 2:])
        extreme=activity_extremization[ac]
        weight=activity_weights[ac]

        clusters_activities.update({ac:{'name': ac, 'model':model, 'clusters':hidden_states, 'values':pivoted_data[ac], 'dates':pivoted_data['interval_end'], 'extremization':extreme, 'weight':weight}})
    return clusters_activities
'''
########################################################################################
MAPPING CLUSTERS TO GRADES
########################################################################################
'''
#Creates map (dictionary) between clusters (cluster numbers) and their means
def create_map__means_to_clusters(model):
    means = [model.means_[i][0] for i in range(len(model.means_))]
    clusters=list(range(len(model.means_)))
    return dict(zip(clusters, means))

# Creates map (dictionary) from cluster means to grades 1-5. Based on extremization type, sorts means and assigns grades
def create_map_means_to_grades(model, activity, activity_extremization):
    extrem=activity_extremization
    means=[model.means_[i][0] for i in range(len(model.means_))]
    sorted = np.sort(means)
    if extrem == 'max':
        grades=range(1, len(sorted)+1)
    else:
        grades = range(len(sorted),0,-1)
    return(dict(zip(grades, sorted)))

# Creates map from clusters to grades - this map is input for mapping all time points (clusters) to grades
def create_map_clusters_to_grades(map_grades, map_clusters):
    map_clusters_grades={}
    for key_clust, value_clust in map_clusters.items():
        for key_grade, value_grade in map_grades.items():
            if value_grade == value_clust:
                map_clusters_grades.update({key_clust:key_grade})
    return(map_clusters_grades)


### Assigns grades to each cluster and returns map
def map_grades_to_clusters(clusters, map_clusters_grades):
    grades=list(map(lambda x: map_clusters_grades[x], clusters))
    return(grades)

### Calculates grade for each time point based on single variate clusters
### Only this method should be called
def calculate_grades(res):
    activities=res.keys()
    for activity in activities:
        model=res[activity]['model']
        activity=res[activity]['name']
        clusters=res[activity]['clusters']
        activity_extremization=res[activity]['extremization']
        map_grades = create_map_means_to_grades(model, activity, activity_extremization)
        map_clusters = create_map__means_to_clusters(model)
        map_clusters_grades = create_map_clusters_to_grades(map_grades, map_clusters)
        grades = map_grades_to_clusters(clusters, map_clusters_grades)
        res[activity].update({'grades':grades})


### Creates higher factor based on result object
def create_higher_factor(res):
    activities = res.keys()
    key_for_len=list(activities)[0]
    size=len(res[key_for_len]['grades'])
    factor=np.zeros(size)
    for activity in activities:
        weight=res[activity]['weight']
        grades=res[activity]['grades']
        weighted_grades=np.multiply(weight,grades)
        factor=factor+weighted_grades
    return (factor)


'''
################################################################## 
WEIGHT CALCULATION - ALTERNATIVE WAYS TO CALCULATE WEIGHTS BETWEEN SUBFACTORS AND FACTORS
#######################################################################
'''
#Calculates weights based on PCA
#All PCAs are computed for single sub-factor
#All sub-factors are normalized based on components
#All sub-factors are weighted by explained variance of each component
def calculate_measure_weights(pivoted_data):
    from sklearn.decomposition import PCA
    X=pivoted_data.iloc[:,2:]
    pca = PCA(n_components=4)
    pca.fit(X)

    components=np.matrix(np.abs(pca.components_)).transpose()
    normalizer=components.sum(axis=0)
    components=np.divide(components, normalizer)
    weights=np.dot(components,pca.explained_variance_ratio_)
    return weights
'''
##########################################################################
VIZUALIZATION
###########################################################################
'''
### plots clusters for single activity (one plot)
def plot_single_variate_cluster(activity, model, hidden_states, values, dates):
    ### Subplot the states multi-variate single user - By States
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
###########################################################################
PROGRAM LOGIC
###########################################################################
'''

##### For single user and each activity -
# 5 clusters,
data = pd.read_csv('activities_out.csv') # Reads data with original and normalized values for each user and activity

user=66 # selects one user id
activities=['sleep_awake_time','sleep_deep_time', 'sleep_light_time', 'sleep_tosleep_time', 'sleep_wakeup_num']
activity_extremization = {'sleep_light_time':'max', 'sleep_deep_time':'max', 'sleep_awake_time':'min', 'sleep_wakeup_num':'min', 'sleep_tosleep_time':'min'}
activity_weights = {'sleep_light_time':0.1, 'sleep_deep_time':0.3, 'sleep_awake_time':0.1, 'sleep_wakeup_num':0.3, 'sleep_tosleep_time':0.2}

res=create_single_variate_clusters(data, activities, activity_extremization, activity_weights)
plot_single_variate_multiple_activities(res)

calculate_grades(res)

factor=create_higher_factor(res)

#### TODO: Different types of aggregation on monthly level
### nicer graph for factors
### group factor on monthly level - this is started
### Try different grades - PCA and variations

res['sleep_deep_time']['grades']

plot_single_series('motility', factor, res['sleep_deep_time']['dates'])

plt.plot(dates,factor)

type(dates)
fac=pd.Series(factor, index=list(range(len(factor))), name='factor')

dates_grades=pd.concat([dates, fac], axis=1)

pd.groupby(dates_grades, by=[dates_grades.interval_end.month, dates_grades.interval_end.year])['mean']





dates_grades['year'] = [y.year for y in dates_grades['interval_end']]
dates_grades['month'] = [m.month for m in dates_grades['interval_end']]
dates_grades['day'] = [d.day for d in dates_grades['interval_end']]


