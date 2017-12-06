from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn import pipeline
from sklearn import model_selection
from sklearn import metrics
import pandas as pd


#FEATURE SELECTION




linear_regression=linear_model.LinearRegression()


# random forest parameters

random_forest=RandomForestRegressor(n_estimators=30, max_depth=3)


dataf=pd.read_csv('Aripiprazol.csv')

Y=dataf.iloc[:,-1]
X=dataf.iloc[:,:-1]

scoring = ['explained_variance','neg_mean_absolute_error', 'neg_mean_squared_error', 'r2', 'neg_mean_squared_log_error', 'neg_median_absolute_error']




linear_regression.fit(X, Y)

cv=model_selection.cross_validate(random_forest.fit(X,Y),X,Y,cv=10, scoring=scoring, return_train_score=True)




print(cv)