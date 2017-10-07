### read the data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

data = pd.read_csv('D:\\Posao\\Projekti\\City4Age\\activities_out.csv')
list(data) # check attribute names

#### Normalization

#### Filter
# filter only attributes of needs
cols = [ 'detection_variable_type',
 'detection_variable_name',
 'zero_time',
 'interval_start',
 'interval_end',
 'step',
 'measure_value',
 'Normalised']

###walk_distance
oneexample = data[data['user_in_role_id'] ==  66] # get one user
oneexample = oneexample[cols]
oneexample.detection_variable_name.unique() # get distinct values of activities
oneexample = oneexample[oneexample['detection_variable_name'] =='walk_distance'] # select one activity
oneexample.head(n= 10)

Y = oneexample[['Normalised']]
X = oneexample[['step']]


kmeans = KMeans(n_clusters=3)
kmeansoutput = kmeans.fit(Y)
kmeansoutput

kmeansoutput.labels_


colormap = np.array(['red', 'lime','black'])
plt.figure(figsize=(14, 7))
# Plot the Models Classifications
plt.scatter(X.step, oneexample.measure_value, c=colormap[kmeans.labels_], s=40)
plt.title('K Mean Classification')

## physicalactivity_calories
###
oneexample = data[data['user_in_role_id'] ==  66] # get one user
oneexample = oneexample[cols]
oneexample.detection_variable_name.unique() # get distinct values of activities
oneexample = oneexample[oneexample['detection_variable_name'] =='physicalactivity_calories'] # select one activity
oneexample.head(n= 10)

Y = oneexample[['Normalised']]
X = oneexample[['step']]


kmeans = KMeans(n_clusters=3)
kmeansoutput = kmeans.fit(Y)
kmeansoutput

kmeansoutput.labels_

import numpy as np
data
from hmmlearn import hmm
Y=data[['step', 'Normalised']]
model2 = hmm.GaussianHMM(3, "full", )
model2.fit(Y)

model2

type(Y.transpose())
colormap = np.array(['red', 'lime','black'])
plt.figure(figsize=(14, 7))
# Plot the Models Classifications
plt.scatter(X.step, oneexample.measure_value, c=colormap[kmeans.labels_], s=40)
plt.title('K Mean Clusters')


##################### GaussianMixture ###############################
gm = GaussianMixture(n_components=3,covariance_type='full').fit(Y)
gmoutput= gm.predict(Y)
gmoutput_prob= gm.predict_proba(Y)

plt.figure(figsize=(14, 7))
# Plot the Models Classifications
colormap = np.array(['red', 'lime','black'])
plt.scatter(X.step, oneexample.measure_value, c=colormap[gmoutput], s=40)
plt.title('GMM Clusters')

### Test again without pull req

np.random.seed(42)

model = hmm.GaussianHMM(n_components=3, covariance_type="full")
model.startprob_ = np.array([0.6, 0.3, 0.1])
model.transmat_ = np.array([[0.7, 0.2, 0.1],
                             [0.3, 0.5, 0.2],
                             [0.3, 0.3, 0.4]])
model.means_ = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]])
model.covars_ = np.tile(np.identity(2), (3, 1, 1))
X, Z = model.sample(100)