import numpy as np
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import utils

def get_data(num_data_points, w0=2.0, b0=1.0, frac_for_test=0.2):
    X, y, y_actual = make_synthetic_data_for_linear_regression(num_data_points, w0, b0)
    X_tr, y_tr, X_te, y_te = split_data_into_train_and_test_sets(X,y, frac_for_test)
    return X_tr, y_tr, X_te, y_te


def make_synthetic_data_for_linear_regression(num_data_points=10, w=2.0, b=1.0):
    noise = stats.norm.rvs(loc=0, scale=1, size=num_data_points)
    xmin = stats.norm.ppf(0.01) # at 1% probability
    xmax = stats.norm.ppf(0.99)# at 90% probability
    X = np.linspace(xmin, xmax, num_data_points)
    y_actual = b + w * X
    y = b + w * X + noise
    return (X,y, y_actual)

def view_X_y_scatter(X,y, y_actual=None):
    plt.scatter(X,y, facecolors='w', edgecolors='k', label="data")
    if y_actual is not None:
        plt.plot(X,y_actual, '-r', lw=3, label="actual")
    plt.show()

def split_data_into_train_and_test_sets(X,y, frac_for_test=0.2):
    n = len(X)
    n_train = int(n * (1-frac_for_test))
    indices_for_training = np.random.choice(range(n), size=n_train, replace=False)
    indices_for_testing = [i for i in range(n) if i not in indices_for_training]
    X_tr, y_tr = X[indices_for_training], y[indices_for_training]
    X_te, y_te = X[indices_for_testing], y[indices_for_testing]
    return X_tr, y_tr, X_te, y_te