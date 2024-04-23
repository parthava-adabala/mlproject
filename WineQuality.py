import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

np.random.seed(1)
df = pd.read_csv('winequality-red.csv')
X = df.drop(columns=['quality'])
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

# start to write your code
trees_values = [i for i in range(100, 501, 5)]
mses = []
for n_trees in trees_values:  # iterate over tree values to find the best no of trees
    model = RandomForestRegressor(n_estimators=n_trees, max_depth=1000, max_features='sqrt', n_jobs=-1, oob_score=True,
                                  random_state=42)
    # create a random forest regressor with each n_trees value
    model.fit(X_train, y_train)  # fit the model with training data
    mse = (1 - model.oob_score_) * np.var(y_train)  # find the mse for every n_trees
    mses.append(mse)  # store the mse in mses list
min_mse_idx = np.argmin(mses)  # get the index of min mse
B = trees_values[min_mse_idx]  # B is the best no of trees
print("Best no of trees: ", B)
# create and fit the best model with B trees
best_model = RandomForestRegressor(n_estimators=B, max_depth=1000, max_features='sqrt', n_jobs=-1, oob_score=True,
                                   random_state=42)
best_model.fit(X_train, y_train)
mse = (1 - best_model.oob_score_) * np.var(y_train)  # get the mse
print("Best OOB MSE: ", mse)

# feature importance
print("Features importance:")
for i in range(len(X.columns)):
    print(X.columns[i], ":", best_model.feature_importances_[i])

# predictions for y
y_pred = best_model.predict(X_test)

# horizontal bar plot for OOB MSE vs. #trees
plt.barh(trees_values, mses)
plt.title("OOB MSE vs. #trees")
plt.xlabel("OOB MSE")
plt.ylabel("#trees")
plt.show()


def adaboost(X_train, y_train, regressor, n, X_test):
    w = np.ones(len(X_train)) / len(X_train)
    y_pred_ada = np.zeros(len(X_train))
    for i in range(n):
        regressor.fit(X_train, y_train, sample_weight=w)
        y_pred_i = regressor.predict(X_train)
        err = np.sum(w * (y_pred_i - y_train) ** 2) / np.sum(w)
        alpha = 0.5 * np.log((1 - err) / max(err, 1e-10))
        w = w * np.exp(-alpha * y_train * y_pred_i)
        w = w / np.sum(w)
        y_pred_ada += alpha * y_pred_i
    return pd.Series(y_pred_ada)


tree_model = DecisionTreeRegressor(max_depth=1000)
y_pred_adaboost = adaboost(X_train, y_train, regressor=tree_model, n=10, X_test=X_test)
random_forest_mse = mean_squared_error(y_test, y_pred)
adaboost_mse = mean_squared_error(y_test, y_pred_adaboost)
print("Random Forest OOS MSE: ", random_forest_mse)
print("AdaBoost OOS MSE: ", adaboost_mse)
