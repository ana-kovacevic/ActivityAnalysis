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

#dp.get_users_activities(data,66)
#pp=dp.prepare_data(data,66,['sleep_deep_time'])

users=[66,67,68]
#####Persist models

hmm_opt.create_single_variate_clusters(66, ['sleep_light_time', activities, activity_extremization, activity_weights)

def create_dict_activities_means_covars(user, activities, model):
    '''    
    :param model: HMM model
    :return:
     Creates dict of the Hmm model for persistance in JSON
    '''
    means = model.means_
    covars = model.covars_
    transmat = model.transmat_

    dict = {}
    for ac in zip (activities, means, covars):
        dict.update({ac[0]:{'means':means[0].tolist(),'covars':covars[0].tolist()}})
    dict.update({'transmat': transmat})
    dict={user:dict}

    return dict



import pickle



a=load_pickle_hmm(66, "sleep_deep_time")


model=joblib.load(path + '/' + filename + '.pkl')




import model_persistance as mp
dict=mp.create_dict_activities_means_covars(user,activities,model)

dict.keys()















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