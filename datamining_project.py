import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import mean
import math
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn import preprocessing

## Begin with loading weather data
data = pd.read_csv('knmi.csv')

## Adding weekdays as column to the data
weekdays = []
for i in range(208):
  weekdays.append("Fri")
  weekdays.append("Sat")
  weekdays.append("Sun")
  weekdays.append("Mon")
  weekdays.append("Tue")
  weekdays.append("Wed")
  weekdays.append("Thu")
weekdays.append("Fri")
weekdays.append("Sat")
weekdays.append("Sun")
weekdays.append("Mon")
weekdays.append("Tue")
data['weekday'] = weekdays

## Adding month as column to the data from the dates already present in the data
months = []
# Loop through rows of data
for key, value in data.iterrows(): 
  # Get from the second column of every row the characters 5 - 7 (= the month)
  months.append((str(value[1])[4:6]))
data['month'] = months

## Preprocessing traffic
traffic = pd.read_csv('traffic.csv', sep=';')
traffic['DatumFileBegin'] = traffic['DatumFileBegin'].str.replace('-', '')
traffic['FileZwaarte'] = traffic['FileZwaarte'].str.replace(',', '.')
## Adding filezwaarte as a column to the previously defined data Dataframe
filezwaarte_per_day = []
# Loop through the rows of the data Dataframe
for data_rows in data.itertuples():
  # Save all the traffic jams from a particular day to a list
  traffic_per_day = traffic.loc[traffic['DatumFileBegin'] == str(data_rows.YYYYMMDD)]
  # Sum all the traffic jams to one value and add this to the list that will be the new column
  filezwaarte_per_day.append(traffic_per_day['FileZwaarte'].astype(float).sum())
data['filezwaarte'] = filezwaarte_per_day

## Transforming categorical variables by giving each of the possible categorial values a seperate binary column (the value is either true or false)
data = pd.get_dummies(data)

## Create y numpy array
y = np.array(data['filezwaarte'])
## Create X numpy array
X = data.drop('filezwaarte', axis = 1)
feature_names_list = list(X.columns)
X = np.array(X)
# Scaling because otherwise the SVR has a too long runtime
X = preprocessing.scale(X)



### REGRESSION ALGORITHMS

## Multiple linear regression (cross validated)
from sklearn.linear_model import LinearRegression
kf = KFold(n_splits=3, random_state=42, shuffle=True)
scores = []
for train_index, test_index in kf.split(X):
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  reg = LinearRegression().fit(X_train, y_train)
  preds = reg.predict(X_test)
  scores.append(math.sqrt(mean_squared_error(preds, y_test)))
print(mean(scores))

## Random Forest (double cross validated)
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  StandardScaler
# Initialize a new Random Forest Regressor
rf = RandomForestRegressor(random_state = 42)
# Split the data in 3 parts using kFold
cv_outer = KFold(n_splits=3, random_state=42, shuffle =True)
outer_results_dcv_rf = []
# Loop through all the splits
for train_index, test_index in cv_outer.split(X):
    # Divide data into test and train data
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # Split data in 4 parts using kFold
    cv_inner = KFold(n_splits=4, shuffle=True, random_state=42)
    # Define the possible parameters-values that will be tested
    param_grid = dict(
        n_estimators=[10, 25, 50, 100],
        max_depth=[20, 30,50],
        min_samples_leaf=[1,2,4])
    # Create the gridSearch
    search = GridSearchCV(rf, param_grid, scoring='neg_root_mean_squared_error', cv=cv_inner)
    pipe = Pipeline([('scaler', StandardScaler()), ('model', search)])
    # Fit the gridSearch on the training data
    result = pipe.fit(X_train, y_train)
    # Take the best estimator out of the fitted data
    # Predict the test data with the use of the best estimator
    y_pred = result.predict(X_test)
    # Compute the error
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # Add the error to the outer results
    outer_results_dcv_rf.append(rmse)
    # Print errors, best scores and best values
    print('>rmse=%.3f' % (rmse))
print(mean(outer_results_dcv_rf))

## Random Forest hyperparameter tuning
# Split data in 4 parts using kFold
cv_inner = KFold(n_splits=4, shuffle=True, random_state=42)
# Define the possible parameters-values that will be tested
param_grid = dict(
    n_estimators=[10, 25, 50, 100],
    max_depth=[20, 30,50],
    min_samples_leaf=[1,2,4])
# Create the gridSearch
search = GridSearchCV(rf, param_grid, scoring='neg_root_mean_squared_error', cv=cv_inner)
# Fit the gridSearch on the training data
result = search.fit(X, y)
print(search.best_params_)

## Support Vector Regression (double cross validated)
from sklearn.svm import SVR
# Initialize a new Support Vector Regressor
svr = SVR(cache_size=7000)
# Split the data in 3 parts using kFold
cv_outer = KFold(n_splits=3, random_state=42, shuffle=True)
outer_results_dcv_svr = []
# Loop through all the splits
for train_index, test_index in cv_outer.split(X):
    # Divide data into test and train data
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # Split data in 4 parts using kFold
    cv_inner = KFold(n_splits=4, shuffle=True, random_state=42)
    # Define the possible parameters-values that will be tested
    param_grid = dict(
        C=[0.1,1.0,10,100],
        gamma=[0.0001,0.01,0.1,1,10],
        kernel= ['rbf', 'poly', 'sigmoid', 'linear'])
    # Create the gridSearch
    search = GridSearchCV(svr, param_grid, scoring='neg_root_mean_squared_error', cv=cv_inner)
    # Fit the gridSearch on the training data
    result = search.fit(X_train, y_train)
    # Take the best estimator out of the fitted data
    best = result.best_estimator_
    # Predict the test data with the use of the best estimator
    y_pred = best.predict(X_test)
    # Compute the error
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # Add the error to the outer results
    outer_results_dcv_svr.append(rmse)
    # Print errors, best scores and best values
    print('>rmse=%.3f, best_score=%.3f, best parameter values=%s' % (rmse, result.best_score_, result.best_params_))
print(mean(outer_results_dcv_svr))

## Support Vector Regression hyperparameter tuning
# Split data in 4 parts using kFold
cv_inner = KFold(n_splits=4, shuffle=True, random_state=42)
# Define the possible parameters-values that will be tested
param_grid = dict(
    C=[0.1,1.0,10,100],
    gamma=[0.0001,0.01,0.1,1,10],
    kernel= ['rbf', 'poly', 'sigmoid', 'linear'])
# Create the gridSearch
search = GridSearchCV(svr, param_grid, scoring='neg_root_mean_squared_error', cv=cv_inner)
# Fit the gridSearch on the training data
result = search.fit(X, y)
print(search.best_params_)

## XGBoost (double cross validated)
import xgboost as xgb
# Initialize a new XGBRegressor
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', booster = 'gbtree', learning_rate = 0.1)
# Split the data in 3 parts using kFold
cv_outer = KFold(n_splits=3, random_state=42, shuffle =True)
outer_results_dcv_xgb = []
# Loop through all the splits
for train_index, test_index in cv_outer.split(X):
  # Divide data into test and train data
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # Split data in 4 parts using kFold
    cv_inner = KFold(n_splits=4, shuffle=True, random_state=42)
    # Define the possible parameters-values that will be tested
    param_grid = dict(
        min_child_weight = [3, 5, 7],
        gamma = [0, 0.01, 0.1],
        subsample = [0.6, 0.7, 0.9],
        colsample_bytree = [0.6, 0.7, 0.8],
        max_depth = [3, 4, 5])
    # Create the gridSearch
    search_test1 = GridSearchCV(estimator = xgb.XGBRegressor(objective ='reg:squarederror', booster = 'gbtree', learning_rate = 0.1), 
                      param_grid = param_grid, scoring='neg_root_mean_squared_error',n_jobs=4, cv=cv_inner)
    # Fit the gridSearch on the training data
    result = search_test1.fit(X_train, y_train)
    # Take the best estimator out of the fitted data
    best = result.best_estimator_
    # Predict the test data with the use of the best estimator
    y_pred = best.predict(X_test)
    # Compute the error
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # Add the error to the outer results
    outer_results_dcv_xgb.append(rmse)
    #Print errors, best scores and best values
    print('>rmse=%.3f, best_score=%.3f, best parameter values=%s' % (rmse, result.best_score_, result.best_params_))
print(mean(outer_results_dcv_xgb))

## XGBoost hyperparameter tuning
# Split data in 4 parts using kFold
cv_inner = KFold(n_splits=4, shuffle=True, random_state=42)
# Define the possible parameters-values that will be tested
param_grid = dict(
    min_child_weight = [3, 5, 7],
    gamma = [0, 0.01, 0.1],
    subsample = [0.6, 0.7, 0.9],
    colsample_bytree = [0.6, 0.7, 0.8],
    max_depth = [3, 4, 5])
# Create the gridSearch
search_test1 = GridSearchCV(estimator = xgb.XGBRegressor(objective ='reg:squarederror', booster = 'gbtree', learning_rate = 0.1), 
                  param_grid = param_grid, scoring='neg_root_mean_squared_error',n_jobs=4, cv=cv_inner)
# Fit the gridSearch on the training data
result = search_test1.fit(X, y)
print(search_test1.best_params_)



## Learning curve
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

# Determine the scores
train_sizes, train_scores, validation_scores = learning_curve(
estimator = LinearRegression(),
X = X,
y = y, train_sizes = [50, 80, 200, 500, 700, 974], cv = 3, random_state = 42, shuffle=True,
scoring = 'neg_root_mean_squared_error')
# Make the scores positive
train_scores_mean = -train_scores.mean(axis = 1)
validation_scores_mean = -validation_scores.mean(axis = 1)
# Plot the scores
plt.style.use('seaborn')
plt.plot(train_sizes, train_scores_mean, label = 'Training error')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
plt.ylabel('RMSE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves for a linear regression model', fontsize = 18, y = 1.03)
plt.legend()

# Visualize the data in a boxplot
plt.figure()
plt.boxplot(y)
plt.title("Boxplot of the file-zwaarte")
plt.ylabel("file-zwaarte")
plt.show()