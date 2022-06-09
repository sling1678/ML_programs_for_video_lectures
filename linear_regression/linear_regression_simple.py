import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import utils, data

import sys
import time


## MODEL
class SimpleLinearRegressionModel(object):
    def __init__(self):
        self.w = 0
        self.b = 0
        self.name="Linear Regression"
    def forward(self, X):
        return self.w * X + self.b
    def train(self, X, y, small=1.0e-6):
        n = len(X)

        xAv = np.mean(X)
        yAv = np.mean(y)

        Sxy_times_N_minus1 = np.sum((X-xAv) * (y-yAv))
        Sx_sq_times_N_minus1 = np.sum((X-xAv) * (X-xAv))
        if Sx_sq_times_N_minus1 == 0:
            Sx_sq_times_N_minus1 = small # to prevent divide by zero error
        self.w = Sxy_times_N_minus1/Sx_sq_times_N_minus1
        self.b = yAv - self.w *xAv

    def test(self, X, y, eval_func):
        """
        Test the model using evaluation function, eval_func
        """
        y_pred = self.forward(X)
        return eval_func(y_pred, y)

# Run
def run_one_sample(num_data_points=500, w0=2.0, b0=1.0, frac_for_test=0.2):

    X_tr, y_tr, X_te, y_te = data.get_data(num_data_points, w0, b0, frac_for_test)
    model = SimpleLinearRegressionModel()
    model.train(X_tr, y_tr)
    err_in, err_out = utils.compute_in_sample_and_out_sample_errors(model, utils.root_mean_squared_error, X_tr, y_tr, X_te, y_te)

    if False:
        utils.plot_all(w0, b0, model.w, model.b, err_in, err_out, X_tr, y_tr, X_te, y_te, model_name=model.name, xlabel="X", ylabel="y")
    return len(X_tr), err_in, err_out, model

# Learning Curve
def generate_learning_curve(w0=2.0, b0=1.0, frac_for_test=0.2, sample_sizes=None):
    if sample_sizes is None:
        sample_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150,  200, 250,  300, 400, 500, 600, 700, 800, 900, 1000]
    in_sample_errors = []
    out_sample_errors = []
    training_sample_sizes = []
    start = time.time()

    sys.stdout.write(f"We have {len(sample_sizes)} samples: (at, size) \n")
    for j,sample_size in enumerate(sample_sizes):
        ins, outs = [], []
        for k in range(1000):
            num_training_samples, ei, eo, _ = run_one_sample(sample_size, w0, b0, frac_for_test)
            ins.append(ei)
            outs.append(eo)

        training_sample_sizes.append(num_training_samples)
        in_sample_errors.append(np.mean(np.array(ins)))
        out_sample_errors.append(np.mean(np.array(outs)))
        sys.stdout.write(f"{j+1}:{sample_size}-")

    sys.stdout.write("\n")
    end = time.time()
    print(f"Took {end-start:.3} seconds.")
    fig, ax = plt.subplots()
    fig.figsize = (10,5)
    ax.plot(training_sample_sizes, out_sample_errors, label=r"$E_{out}$")
    ax.plot(training_sample_sizes, in_sample_errors, label=r"$E_{in}$")
    ax.set_title("Learning Curve")
    ax.set_xlabel(r"Number of Training Samples")
    ax.set_ylabel(r"Root Mean Squared Error")
    plt.legend()
    plt.savefig("learning_curve.png")
    plt.show()


if __name__ == "__main__":

    DEBUGGING = False
    
    if DEBUGGING:
        np.random.seed(42)
        n_tr, err_in, err_out, model = run_one_sample(100)

    generate_learning_curve()
