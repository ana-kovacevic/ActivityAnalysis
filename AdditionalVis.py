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
dates=oneexample['interval_end']

#### Cumsum
tsY = pd.DataFrame(Y['Normalised'])
tsY['Date'] = dates
selY = tsY.ix[:].dropna()

selY['Date'] = pd.to_datetime(selY['Date'])
selY.set_index('Date', inplace=True)
selY.info()
colors = cm.rainbow(np.linspace(0, 1, model.n_components))


sns.set(font_scale=1.5)
style_kwds = {'xtick.major.size': 3, 'ytick.major.size': 3,
              'font.family':u'courier prime code', 'legend.frameon': True}
sns.set_style('white', style_kwds)

stateY = (pd.DataFrame(hidden_states, columns=['state'], index=selY.index)
          .join(selY, how='inner')
          .assign(vol_cret=selY.Normalised.cumsum())
          .reset_index(drop=False)
          .rename(columns={'index':'Date'}))
print(stateY.head())

sns.set_style('white', style_kwds)
order = [0, 1, 2]
fg = sns.FacetGrid(data=stateY, hue='state', hue_order=order, palette=colors, aspect=1.31, size=12)
fg.map(plt.scatter, 'Date', 'vol_cret', alpha=0.8).add_legend()

sns.despine(offset=10)
fg.fig.suptitle('Walking Distance states', fontsize=24, fontweight='demi')
#fg.savefig('Hidden Markov (Mixture) Model_SPY Regimes.png')


######## One plot time series

ts = pd.DataFrame(Y['Normalised'])
ts['Date'] = dates
sel = ts.ix[:].dropna()

sel['Date'] = pd.to_datetime(sel['Date'])
sel.set_index('Date', inplace=True)
sel.info()
colors = cm.rainbow(np.linspace(0, 1, model.n_components))


sns.set(font_scale=1.5)
style_kwds = {'xtick.major.size': 3, 'ytick.major.size': 3,
              'font.family':u'courier prime code', 'legend.frameon': True}
sns.set_style('white', style_kwds)

state = (pd.DataFrame(hidden_states, columns=['state'], index=sel.index)
          .join(sel, how='inner')
          .assign(nor_cret=sel.Normalised.cumsum())
          .reset_index(drop=False)
          .rename(columns={'index':'Date'}))
print(state.head())

sns.set_style('white', style_kwds)
order = [0, 1, 2]
fg = sns.FacetGrid(data=state, hue='state', hue_order=order, palette=colors, aspect=1.31, size=12)
fg.map(plt.scatter, 'Date', 'Normalised', alpha=0.8).add_legend()

sns.despine(offset=10)
fg.fig.suptitle('Walking Distance states', fontsize=24, fontweight='demi')
#fg.savefig('Hidden Markov (Mixture) Model_SPY Regimes.png')


# simulate data
# =============================

df = selY
df['state'] = hidden_states

# plot
# =============================


fig, ax = plt.subplots()

def plot_func(group):
    global ax
    color = 'r' if (group['state'] == 0).all()  else 'g' if (group['state'] == 1).all() else 'b'
    lw = 2.0
    ax.plot(group.index, group.measure_value, c=color, linewidth=lw)

#df.groupby((df['state'].shift() * df['state'] <= 1).cumsum()).apply(plot_func)

df.apply(plot_func)
df