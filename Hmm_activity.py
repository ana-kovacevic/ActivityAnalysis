from hmmlearn.hmm import GaussianHMM
import pandas as pd
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, DayLocator
import numpy as np

data = pd.read_csv('D:\\Posao\\Projekti\\City4Age\\activities_out.csv')
oneexample = data[data['user_in_role_id'] ==  66] # get one user
oneexample = oneexample[oneexample['detection_variable_name'] =='walk_distance'] # select one activity

oneexample
Y = oneexample[['Normalised']]
X = oneexample[['measure_value']]

type([X])

# Make an HMM instance and execute fit
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


fig, axs = plt.subplots(model.n_components, sharex=True, sharey=True)
colours = cm.rainbow(np.linspace(0, 1, model.n_components))
for i, (ax, colour) in enumerate(zip(axs, colours)):
    # Use fancy indexing to plot data in each state.
    mask = hidden_states == i
    ax.plot_date(dates[mask], X[mask], ".-", c=colour)
    ax.set_title("{0}th hidden state".format(i))

    # Format the ticks.
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_minor_locator(MonthLocator())
    ax.xaxis.set_minor_locator(DayLocator())

    ax.grid(True)

plt.show()