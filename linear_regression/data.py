import numpy as np
import pandas as pd
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

def get_elem_stat_learn_prostate_data():
    """
    Prostate data info

        Predictors (columns 1--8)

        lcavol
        lweight
        age
        lbph
        svi
        lcp
        gleason
        pgg45

        outcome (column 9)

        lpsa

        train/test indicator (column 10)

        This last column indicates which 67 observations were used as the 
        "training set" and which 30 as the test set, as described on page 48
        in the book.

        There was an error in these data in the first edition of this
        book. Subject 32 had a value of 6.1 for lweight, which translates to a
        449 gm prostate! The correct value is 44.9 gm. We are grateful to
        Prof. Stephen W. Link for alerting us to this error.

        The features must first be scaled to have mean zero and  variance 96 (=n)
        before the analyses in Tables 3.1 and beyond.  That is, if x is the  96 by 8 matrix
        of features, we compute xp <- scale(x,TRUE,TRUE)
    """
    X_column_names =["lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45"]
    y_column_name = ["lpsa"]
    path = "https://hastie.su.domains/ElemStatLearn/datasets/prostate.data"
    df = pd.read_csv (path, sep="\t", skiprows=0, index_col=0)
    X_df=df[X_column_names + ["train"]]
    y_df=df[y_column_name + ["train"]]

    X_tr = X_df.loc[X_df['train'] == 'T'].drop("train", axis=1).values
    X_te = X_df.loc[X_df['train'] != 'T'].drop("train", axis=1).values
    y_tr = y_df.loc[y_df['train'] == 'T'].drop("train", axis=1).values
    y_te = y_df.loc[y_df['train'] != 'T'].drop("train", axis=1).values

    if False:
        print(f"Training samples: X shape is {X_tr.shape}, and y shape is {y_tr.shape}")
        print(f"Test samples: X shape is {X_te.shape}, and y shape is {y_te.shape}")
    
    return X_tr, y_tr, X_te, y_te, X_column_names #OK

class Normalizer(object):
    def __init__(self):
        self.mu = None
        self.sigma = None
    
    def fit(self, X):
        self.mu = np.mean(X, axis=0)
        self.sigma = np.std(X, axis=0)
        

    def transform(self, X):
        mu = np.expand_dims(self.mu, axis=0)
        sigma = np.zeros_like(self.sigma)
        for i,s in enumerate(self.sigma):
            if s==0:
                s = 1.0e-6
            sigma[i]=s
        sigma = np.expand_dims(sigma, axis=0) 
        X_normed = (X-mu)/sigma
        return X_normed

def sample_dataset_from_another_dataset(X,y, size:int):
    n = len(X)
    samples = np.random.choice(range(n), size=size, replace=False)
    return X[samples], y[samples]



if __name__ == "__main__":
    
    #get_elem_stat_learn_prostate_data()
    X = np.array([[1,2], [3,4], [5,6]])
    S = Normalizer()
    S.fit(X)
    X1 = S.transform(X)
    print(X1)
    print(np.std(X1, axis=0))
    
 

