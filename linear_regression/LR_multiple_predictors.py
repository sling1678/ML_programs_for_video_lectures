import numpy as np
import pickle
import matplotlib.pyplot as plt
import utils, data

import sys
import time


## Results Directory
RESULTS_DIR = ".\\results"

## Model
class MultiplePredictorsLinReg(object):
    def __init__(self, w=None, b=None):
        self.w = w # a vector
        self.b = b # a scalar
        self.name="Multiple_Predictors_Regression"
    def forward(self, X):
        return np.matmul(X, self.w) + self.b
    
    def train(self, X, y):
        """
        X and y are Numpy ndarrays
        X has shape (N,p); y has shape (N,1)
        """
        try:
            x0 = np.ones(len(X))
            x0 = np.expand_dims(x0, axis=1)
            X_ex = np.concatenate((x0,X), axis=1)
            A = np.matmul(X_ex.T, X_ex) # need to add lambda * eye_like(A) to avoid singular matrix inversion issues - motivation for ridge regression
            identity_matrix = np.eye(len(A), len(A))
            A = A + 10.0 * identity_matrix
            A_inv = np.linalg.inv(A)
            B = np.matmul(A_inv, X_ex.T)
            params = np.matmul(B, y)
            self.b = params[0,0]
            self.w = params[1:,0]
        except np.linalg.LinAlgError: # singular matrix problem without ridge lambda
            raise np.linalg.LinAlgError
 

        if False:
            print(f"X_ex shape = {X_ex.shape}, B shape = {B.shape}")
            print(f"y shape = {y.shape}, params shape = {params.shape}")
    
    def test(self, X, y, eval_func):
        """
        Test the model using evaluation function, eval_func
        """
        y_pred = self.forward(X)
        return eval_func(y_pred, y)

# Run

def run_one_sample(dat, normalize_predictors=True, eval_function=utils.root_mean_squared_error, save_to_file=None, save=False):
    X_tr, y_tr, X_te, y_te, feature_names = dat
    if normalize_predictors:
        S = data.Normalizer()
        S.fit(X_tr)
        X_tr = S.transform(X_tr)
        X_te = S.transform(X_te)
    
    err_in, err_out = None, None
    model = MultiplePredictorsLinReg()
    try:
        model.train(X_tr, y_tr)
        err_in, err_out = utils.compute_in_sample_and_out_sample_errors(model, eval_function, X_tr, y_tr, X_te, y_te)
        if save:
            if save_to_file is None:
                save_to_file = "mult_lin_reg_prostate.pkl"
            obj = {"n_tr":len(X_tr), "n_te":len(X_te), "err_in": err_in, "err_out": err_out, "model":model, "feature_names":feature_names}
            pickle.dump(obj, open(RESULTS_DIR+"\\"+save_to_file, "wb" ))
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError
    
    return len(X_tr), len(X_te), err_in, err_out, model, feature_names

# Learning Curve requires training at multiple numbers of training samples
from collections import defaultdict
def generate_learning_curve(number_of_trials_of_each_size=3, training_on_full_dataset=False):
    
    training_sample_fractions = [0.1*n for n in range(2,11,1)]
    X_tr, y_tr, X_te, y_te, feature_names = data.get_elem_stat_learn_prostate_data()
    training_sample_sizes=[ int(a*len(X_tr)) for a in training_sample_fractions]
    
    start = time.time()
    sys.stdout.write(f"Max {len(X_tr)} samples. Working (at:size) \n")
    in_sample_errors = []
    out_sample_errors = []
    b_as_func_of_size = dict()
    w_as_func_of_size = dict()
    for j,size in enumerate(training_sample_sizes):
        # for each training size perform a number of trials, then average over the results
        errors=defaultdict(list)
        params=defaultdict(list)
        for k in range(number_of_trials_of_each_size):
            X_tr_this, y_tr_this = data.sample_dataset_from_another_dataset(X_tr, y_tr, size)
            dat = X_tr_this, y_tr_this, X_te, y_te, feature_names
            try:
                n_tr, n_te, err_in, err_out, model, feature_names = run_one_sample(dat, save_to_file="mult_lin_reg_learning_curve.pkl", save=False)
            except np.linalg.LinAlgError:
                continue # do the next iteration
            errors["in"].append(err_in)
            errors["out"].append(err_out)
            params["b"].append(model.b)
            params["w"].append(model.w)
        in_sample_errors.append(np.mean(errors["in"]))
        out_sample_errors.append(np.mean(errors["out"]))
        b_as_func_of_size[size] = np.mean(params["b"])
        w_as_func_of_size[size] = np.mean(params["w"], axis=0)
        

        sys.stdout.write(f"({j+1}:{size})-")
    sys.stdout.write("\n")
    end = time.time()
    print(f"Took {end-start:.3} seconds.")
    for k,v in w_as_func_of_size.items():
        sys.stdout.write(f"size = {k}: w's =")
        for wi in v:
            sys.stdout.write(f"{wi:3.2},\t")
        sys.stdout.write("\n")


    fig, ax = plt.subplots()
    fig.figsize = (10,5)
    ax.plot(training_sample_sizes, out_sample_errors, label=r"$E_{out}$")
    ax.plot(training_sample_sizes, in_sample_errors, label=r"$E_{in}$")
    ax.set_title("Learning Curve")
    ax.set_xlabel(r"Number of Training Samples")
    ax.set_ylabel(r"Root Mean Squared Error")
    plt.legend()
    plt.savefig(RESULTS_DIR + "/multi_lin_reg_prostate_learning_curve.png")
    plt.show()


if __name__ == "__main__":

    if False:
        dat = data.get_elem_stat_learn_prostate_data() #*********************
        n_tr, n_te, err_in, err_out, model, feature_names = run_one_sample(dat)
        print(f"n_tr = {n_tr}, n_te = {n_te}, err_in = {err_in:.3}, err_out = {err_out:.3}")
        print("Model parameters:")
        print(f"b = {model.b:.2}")
        trained_coefficients = {k:v for k,v in zip(feature_names, model.w)}
        for k, v in trained_coefficients.items():
            print(f"W_{k} : {v:.2}")
    if False:
        save_to_file="mult_lin_reg_prostate.pkl"
        obj = pickle.load(open(RESULTS_DIR+"\\"+save_to_file, "rb" ))
        n_tr, n_te, err_in, err_out, model, feature_names = obj["n_tr"], obj["n_te"], obj["err_in"], obj["err_out"], obj["model"], obj["feature_names"]
        print("--------------------")
        print(f"n_tr = {n_tr}, n_te = {n_te}, err_in = {err_in:.3}, err_out = {err_out:.3}")
        print("Model parameters:")
        print(f"b = {model.b:.2}")
        trained_coefficients = {k:v for k,v in zip(feature_names, model.w)}
        for k, v in trained_coefficients.items():
            print(f"W_{k} : {v:.2}")

    if True:
        generate_learning_curve(30, False)

        

