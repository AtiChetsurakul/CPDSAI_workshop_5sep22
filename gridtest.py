#1. import the library
from sklearn.model_selection import GridSearchCV
#RandomSearchCV is a faster version

#2. define the search space
#is a dictionary: key(param): value(value)
#go to google and search sklearn RandomForestRegressor, see how many parameters
#what is the idea: no idea - just loop everything
param_grid = {
    'max_depth' : [5, 10],
    'n_estimators' : [5, 6, 7, 8, 9, 10],
    'max_features' : ['auto', 'log2'],
}
#how many combinations? --> 24 combinations

#3. define the model you want to search with
estimator = RandomForestRegressor()

#4. define the gridsearch object with the search space
grid = GridSearchCV(estimator  = estimator,
                    param_grid = param_grid,
                    cv = kfold,
                    n_jobs = -1,
                    refit = True,
                    scoring = 'neg_mean_squared_error')

#5. run the search
grid.fit(X_train,y_train) # why training set
print(grid.best_params_)
print(grid.best_score_)
print(grid.best_estimator_)
