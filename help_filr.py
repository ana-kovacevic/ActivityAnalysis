

#### ARIMA #####

from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(series, order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())

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