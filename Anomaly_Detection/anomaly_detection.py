import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn.ensemble import IsolationForest
import pandas as pd
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, DayLocator


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
    pivoted_data = select_pivot_users_activities(data, user, ac)
    pivoted_data = pivoted_data.reset_index()
    pivoted_data['interval_end'] = pd.to_datetime(pivoted_data['interval_end'])
    pivoted_data = pivoted_data.sort_values(['user_in_role_id', 'interval_end'])
    return pivoted_data


data = pd.read_csv('activities_out.csv')

user=66
activities=['sleep_awake_time','sleep_deep_time', 'sleep_light_time', 'sleep_tosleep_time', 'sleep_wakeup_num']
activity_extremization = {'sleep_light_time':'max', 'sleep_deep_time':'max', 'sleep_awake_time':'min', 'sleep_wakeup_num':'min', 'sleep_tosleep_time':'min'}
activity_weights = {'sleep_light_time':0.1, 'sleep_deep_time':0.3, 'sleep_awake_time':0.1, 'sleep_wakeup_num':0.3, 'sleep_tosleep_time':0.2}


n_samples = 200
outliers_fraction = 0.1
clusters_separation = [0, 1, 2]
iForest=IsolationForest(max_samples=n_samples, contamination=outliers_fraction)


data_prep=prepare_data(data,user, activities)
dates=data_prep['interval_end']
data_prep=data_prep.iloc[:,2:]
iForest.fit(np.array(data_prep))
iForest.predict(data_prep)



type(data_prep)
data_prep.shape[1]
data_prep.head()
###### Visualize anomalies
# Draw vertical line on plot
'''
##########################################################################
VIZUALIZATION
###########################################################################
'''
data_prep[0]
### plots clusters for single activity (one plot)
def plot_multivariate_series(data, dates, activities):
    ### Subplot the states multi-variate single user - By States
    #fig, axs = plt.subplots(model.n_components, sharex=True, sharey=True)
    #colours = cm.rainbow(np.linspace(0, 1, model.n_components))

    colours = cm.rainbow(np.linspace(0, 1, len(activities)))
    #dates=data['interval_end'] # maybe return this line add remove parameter dates
    i=0
    lines=[]

        # Use fancy indexing to plot data in each state.
    #mask = hidden_states == i
    #i=i+1
    for j in range(len(activities)):
        Y = data[activities[j]]
        lines.append(plt.plot_date(dates[j], Y[j], ".-", c=colours[j], label =activities[j]))
    #ax.set_title("{0}th hidden state".format(i))
    # Format the ticks.
    #ax.xaxis.set_major_locator(YearLocator())
        ax = plt.plot_date(dates, Y, ".-", c=colours[j], label=activities[j])
    ax.xaxis.set_minor_locator(MonthLocator())
    ax.xaxis.set_minor_locator(DayLocator())
    ax.grid(True)

        #plt.suptitle("User_in_role_id: " + str(results[0]) + "     Activity: "+str(results[1]))
        #plt.savefig(path_store + 'user_' + str(results[0])+ '_activity_'+str(results[1])+'.png', bbox_inches='tight')
    #fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)
    plt.flatten()[-1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=2)
    plt.show()


activities
plot_multivariate_series(data_prep,dates,activities)



data_prep[activities[4]]
data_prep.head()
a=data_prep['sleep_awake_time']
type(a)



np.random.seed(42)
# Data generation
X1 = 0.3 * np.random.randn(100 // 2, 2) - 2
X2 = 0.3 * np.random.randn(100 // 2, 2) + 2
X = np.r_[X1, X2]
# Add outliers
X = np.r_[X, np.random.uniform(low=-6, high=6, size=(100, 2))]


