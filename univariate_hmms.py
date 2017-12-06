import pandas as pd
import data_preparation as dp
import HMM_Optimization.HMM_optimization as hmm_opt
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, DayLocator
import numpy as np


'''
###########################################################################
PROGRAM LOGIC
###########################################################################
'''

data = pd.read_csv('Data/activities_out.csv') # Reads data with original and normalized values for each user and activity
#dp.get_users_activities(data, 66)


user=66 # selects one user id
activities=['sleep_awake_time','sleep_deep_time', 'sleep_light_time', 'sleep_tosleep_time', 'sleep_wakeup_num']
activity_extremization = {'sleep_light_time':'max', 'sleep_deep_time':'max', 'sleep_awake_time':'min', 'sleep_wakeup_num':'min', 'sleep_tosleep_time':'min'}
activity_weights = {'sleep_light_time':0.1, 'sleep_deep_time':0.3, 'sleep_awake_time':0.1, 'sleep_wakeup_num':0.3, 'sleep_tosleep_time':0.2}




'''ß
Create and store optimal single variate hmm models for each activity
'''

users=[66,67,68]
optimal_hmms_single_variate=[]
subfactor_activities=dp.get_dict_ges_activities()

dp.get_users_activities(data,66)

pp=dp.prepare_data(data,66,['sleep_deep_time'])

for user in users:
    for subfactor, activities in subfactor_activities.items():
        for activity in activities:
            prepared_data = dp.prepare_data(data,user,[activity])
            best_value, best_model = hmm_opt.optimize_number_of_clusters(prepared_data.iloc[: ,2:], list(range(2,11)))
            optimal_hmms_single_variate.append(best_model)

for a in optimal_hmms_single_variate:
    print(a)

len(optimal_hmms_single_variate)
list(range(2,10))
type(a)
a
aa.shape[0]

from hmmlearn.hmm import GaussianHMM

aa=pp.iloc[:, 2:]

model = GaussianHMM(n_components=5, covariance_type="full", n_iter=1000).fit(aa)


a=activities.keys()

hmm_opt.optimize_number_of_clusters(data)


clusters66=hmm_opt.create_single_variate_clusters(data, user, activities, activity_extremization, activity_weights)


dp.get_users_activities(data, 68)


model, pivoted_data, activities, hidden_states=dp.create_multi_variate_clusters(data, user, activities, activity_extremization)

model.means_

import model_persistance as mp
dict=mp.create_dict_activities_means_covars(user,activities,model)

dict.keys()


model=clusters66['sleep_light_time']['model']





##### For single user and each activity -
# 5 clusters,



data['user_in_role_id'].unique()

data.head(2)



data.head(5)


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