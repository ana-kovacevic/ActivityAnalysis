from hmmlearn.hmm import GaussianHMM
import pandas as pd
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, DayLocator
import numpy as np
import seaborn as sns

data = pd.read_csv('D:\\Posao\\Projekti\\activities_out.csv')

oneexample = data[data['user_in_role_id'] ==  66] # get one user
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


#################### Train on  value (X)
model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000).fit(X)

# Predict the optimal sequence of internal hidden state
hidden_states = model.predict(X)

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

#### Subplot of states
fig, axs = plt.subplots(model.n_components, sharex=True, sharey=True)
colours = cm.rainbow(np.linspace(0, 1, model.n_components))
for i, (ax, colour) in enumerate(zip(axs, colours)):
    # Use fancy indexing to plot data in each state.
    mask = hidden_states == i
    ax.plot_date(dates[mask], X[mask], ".-", c=colour)
    ax.set_title("{0}th hidden state".format(i))

    # Format the ticks.
    #ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_minor_locator(MonthLocator())
    ax.xaxis.set_minor_locator(DayLocator())

    ax.grid(True)

plt.show()
############################### Color time series by States
tsX = pd.DataFrame(Y['Normalised'])
tsX['Date'] = dates
selX = tsX.ix[:].dropna()

selX['Date'] = pd.to_datetime(selX['Date'])
selX.set_index('Date', inplace=True)
stateX = (pd.DataFrame(hidden_states, columns=['state'], index=selX.index)
          .join(selX, how='inner')
          .assign(vol_cret=selX.measure_value.cumsum())
          .reset_index(drop=False)
          .rename(columns={'index':'Date'}))
print(stateX.head())

a = stateX['state']
x = stateX.index
y = stateX['measure_value']
for a, x1, x2, y1, y2 in zip(a[1:], x[:-1], x[1:], y[:-1], y[1:]):
    if a == 0:
        plt.plot([x1, x2], [y1, y2], 'r')
    elif a == 1:
        plt.plot([x1, x2], [y1, y2], 'g')
    else:
        plt.plot([x1, x2], [y1, y2], 'b')

plt.show()
