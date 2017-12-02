from hmmlearn.hmm import GaussianHMM
import pandas as pd
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, DayLocator
import numpy as np




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




def optimize_number_of_clusters(data, range_of_clusters):
    '''    
    :param data: 
    :param range_of_clusters: 
    :return: 
    '''

    best_value=np.inf # create for maximization and minimization
    best_model=None
    for n_states in range_of_clusters:
        model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000).fit(data)
        log_likelihood = model.score(data)
        n_features = 1 ### here adapt for multi-variate
        n_params = n_states * (n_states - 1) + 2 * n_features * n_states
        logN = np.log(len(data))
        bic = -2 * log_likelihood + n_params * logN
        if bic < best_value:
            best_value, best_model = bic, model
    return best_value, best_model


'''
###########################################################################
PROGRAM LOGIC
###########################################################################
'''

import data_preparation
import pandas as pd
data = pd.read_csv('activities_out.csv') # Reads data with original and normalized values for each user and activity

from data_preparation import get_users_activities

import data_preparation as dp


dp.get_users_activities(data, 14)




##### For single user and each activity -
# 5 clusters,



data['user_in_role_id'].unique()

data.head(2)



data.head(5)

user=66 # selects one user id
activities=['sleep_awake_time','sleep_deep_time', 'sleep_light_time', 'sleep_tosleep_time', 'sleep_wakeup_num']
activity_extremization = {'sleep_light_time':'max', 'sleep_deep_time':'max', 'sleep_awake_time':'min', 'sleep_wakeup_num':'min', 'sleep_tosleep_time':'min'}
activity_weights = {'sleep_light_time':0.1, 'sleep_deep_time':0.3, 'sleep_awake_time':0.1, 'sleep_wakeup_num':0.3, 'sleep_tosleep_time':0.2}

pivoted_data=prepare_data(data, user, 'sleep_tosleep_time')

train=pivoted_data.iloc[:-30,:]
test=pivoted_data.iloc[-30:,:]

result=optimize_number_of_clusters(train[['sleep_tosleep_time']], [2,3,4,5,6,7,8,9,10])

model=result[1]

predictions=model.predict(test[['sleep_tosleep_time']])

model.means_

def mean_to_state(state):
    if state==0:
        return(0.12079959)
    elif state==1:
        return (0.21201418)
    else:
        return (0.73009568)

results=list(map(mean_to_state, predictions))

from sklearn import metrics

metrics.mean_squared_error(test[['sleep_tosleep_time']],results)

type(test['sleep_tosleep_time'])

type(pd.Series(results))

plt.scatter(test['sleep_tosleep_time'], results)

panda=pd.concat([test['sleep_tosleep_time'], pd.Series(results)], axis=1)

len(y)
len(results)
y=test['sleep_tosleep_time']

fig, ax = plt.subplots()
ax.scatter(y, results, edgecolors=(0, 0, 0))
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


X, Z = model.sample(30)


test_values=list(test['sleep_tosleep_time'])
pd.DataFrame(test_values, results)


test[['sleep_tosleep_time']]






'''
pivoted_data.head()

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


'''