from hmmlearn.hmm import GaussianHMM
import pandas as pd
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, DayLocator
import numpy as np
import datetime as dt

data = pd.read_csv('activities_out.csv')

get_users_activities(data,66)

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

#[ 0.16932597,  0.23696033,  0.2439348 ,  0.11758979,  0.20387557]])


#activities=['sleep_tosleep_time']

pivoted_data=select_pivot_users_activities(data, user, activities)

pivoted_data = pivoted_data.reset_index()
pivoted_data['interval_end']=pd.to_datetime(pivoted_data['interval_end'])

pivoted_data = pivoted_data.sort_values(['user_in_role_id','interval_end'])

model = GaussianHMM(n_components=4, covariance_type="diag", n_iter=1000).fit(pivoted_data.iloc[:,2:])
hidden_states=model.predict(pivoted_data.iloc[:,2:])


dates=(pivoted_data['interval_end']).dt.to_pydatetime()





#### Print model parameters
#maybe add whole covariances
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

def plot_states(model, pivoted_data, activities, hidden_states):
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
    axs.flatten()[-1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=2)
    plt.show()

#for ac in activities:


### This may be better if all data and than select one by one
def prepare_data(data, user, ac):
    pivoted_data = select_pivot_users_activities(data, user, [ac])
    pivoted_data = pivoted_data.reset_index()
    pivoted_data['interval_end'] = pd.to_datetime(pivoted_data['interval_end'])
    pivoted_data = pivoted_data.sort_values(['user_in_role_id', 'interval_end'])
    return pivoted_data


##### For single user and each activity -
# 5 clusters,
user=66
activities=['sleep_awake_time','sleep_deep_time', 'sleep_light_time', 'sleep_tosleep_time', 'sleep_wakeup_num']
activity_extremization = {'sleep_light_time':'max', 'sleep_deep_time':'max', 'sleep_awake_time':'min', 'sleep_wakeup_num':'min', 'sleep_tosleep_time':'min'}
activity_weights = {'sleep_light_time':0.1, 'sleep_deep_time':0.3, 'sleep_awake_time':0.1, 'sleep_wakeup_num':0.3, 'sleep_tosleep_time':0.2}


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

def plot_single_variate_clusters(activity, model, hidden_states, values, dates):
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


def create_map__means_to_clusters(model):
    means = [model.means_[i][0] for i in range(len(model.means_))]
    clusters=list(range(len(model.means_)))
    return dict(zip(clusters, means))


activity='sleep_awake_time'

def create_map_means_to_grades(model, activity, activity_extremization):
    extrem=activity_extremization
    means=[model.means_[i][0] for i in range(len(model.means_))]
    sorted = np.sort(means)
    if extrem == 'max':
        grades=range(1, len(sorted)+1)
    else:
        grades = range(len(sorted),0,-1)
    return(dict(zip(grades, sorted)))


def create_map_clusters_to_grades(map_grades, map_clusters):
    map_clusters_grades={}
    for key_clust, value_clust in map_clusters.items():
        for key_grade, value_grade in map_grades.items():
            if value_grade == value_clust:
                map_clusters_grades.update({key_clust:key_grade})
    return(map_clusters_grades)



def map_grades_to_clusters(clusters, map_clusters_grades):
    grades=list(map(lambda x: map_clusters_grades[x], clusters))
    return(grades)



for cluster, grade in zip(clusters, grades):
    print(cluster, grade)


res=create_single_variate_clusters(data, activities, activity_extremization, activity_weights)
res['sleep_deep_time']['grades']




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

calculate_grades(res)

activity=res['sleep_awake_time']['name']
clusters=res['sleep_awake_time']['clusters']
model=res['sleep_awake_time']['model']
values=res['sleep_awake_time']['values']
dates=res['sleep_awake_time']['dates']

plot_single_variate_clusters(activity, model, hidden_states, values, dates)

means=[model.means_[i][0] for  i in range(len(model.means_))]
clusters_means=dict(zip([0,1,2,3,4], means))


a=np.sort(list(clusters_means.values()))
grades=dict(zip([1,2,3,4,5], a))

cluster_grades = {}
for key_clust, value_clust in clusters_means.items():
    for key_grade, value_grade in grades.items():
        if value_clust == value_grade:
            cluster_grades.update({key_clust:key_grade})

# try with compehension
# try with sorted representation of dict
a=clusters.map(lambda x: cluster_grades[x])

a=map(lambda x: cluster_grades[x], clusters)
for m in a:
   print(m)

b=zip(clusters, a)

list(b)
for ab in b:
    print(ab)


for ab in a:
    print(ab)

for i in range(model.n_components):
    print("{0}th hidden state".format(i))
    print("mean = ", model.means_[i])
    print("var = ", np.diag(model.covars_[i]))
    print()

    return model, pivoted_data, [ac], hidden_states

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



##### PCA ###

### different types of PCAs, evaluation,
##!!!!! solve extremization - everythin to be on max

def calculate_measure_weights():
    from sklearn.decomposition import PCA
    X=pivoted_data.iloc[:,2:]
    pca = PCA(n_components=4)
    pca.fit(X)

    components=np.matrix(np.abs(pca.components_)).transpose()
    normalizer=components.sum(axis=0)
    components=np.divide(components, normalizer)
    weights=np.dot(components,pca.explained_variance_ratio_)
    return weights

calculate_measure_weights()




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