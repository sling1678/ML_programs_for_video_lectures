import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt


def root_mean_squared_error(predicted, actual):
    """
    Error Function to be used here
    """
    if not isinstance(predicted,(list,pd.core.series.Series,np.ndarray)):
        predicted = np.array([predicted])
    if not isinstance(actual,(list,pd.core.series.Series,np.ndarray)):
        actual = np.array([actual])
    if isinstance(predicted, list):
        predicted = np.array(predicted)
    if isinstance(actual, list):
        actual = np.array(actual)
    return np.sqrt( np.sum(np.square(predicted - actual))/len(predicted) )

def compute_in_sample_and_out_sample_errors(model, err_function, X_tr, y_tr, X_te=None, y_te=None):
    """
    in_sample_error = error on the samples used to train the model
    out_sample_error = error on new samples not seen by the model
    """
    err_in, err_out = None, None
    if X_tr is not None and y_tr is not None:
        err_in = model.test(X_tr, y_tr, err_function)
    if X_te is not None and y_te is not None:
        err_out = model.test(X_te, y_te, err_function)
    return err_in, err_out

def plot_linear_regression(w0, b0, w, b, err_in, err_out, X_tr, y_tr, X_te, y_te, model_name="Linear Regression", xlabel="X", ylabel="y", results_directory=None, filename=None):
    """
    Plot all relevant plots in one place.
    """
    plt.scatter(X_tr, y_tr, c='k', alpha=0.3, label='training data')
    plt.scatter(X_te, y_te, c='red', alpha=0.3,label="test data")
    
    plt.plot(X_tr, w0*X_tr + b0, '--b', alpha=0.8, lw=5, label="true")
    plt.plot(X_tr, w*X_tr + b, '-g', lw=3, label="predicted")

    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(model_name)
    plt.text(0.2, -4, f"E_in={err_in:5.2}, E_out={err_out:5.2}")
    plt.text(0.2, -4.8, f"N={len(X_tr)+len(X_te)}, N_tr={len(X_tr)}, N_te={len(X_te)}")
    # save
    if filename is None:
        filename="plot_generated.png"
    if results_directory is None:
        results_directory = "."
    filename = results_directory +"\\" + filename
    plt.savefig(filename)
    plt.show()


def make_plot(x, multiple_ys, lw=3, linestyle=None, labels=None, colors=None):
    fig, ax = plt.subplots(1, 1)
    if linestyle is None:
        if len(multiple_ys)==1:
            linestyle=['-']
        elif len(multiple_ys)==2:
            linestyle=['-', ':']
        elif len(multiple_ys)==3:
            linestyle=['-', ':', '-.']
    if labels is None:
        labels=[]
        for i,_ in enumerate(multiple_ys):
            labels.append("y_"+str(i))
    for i,y in enumerate(multiple_ys):
        if len(multiple_ys)<4:
            ls = linestyle[i]
        else:
            ls = '-'
        ax.plot(x, y, lw=lw, linestyle=ls, label=labels[i])
    ax.legend(loc='best', frameon=False)
    
    plt.show()


