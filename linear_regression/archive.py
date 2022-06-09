import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def sample_data(num_data_points, plotting=False):
    samples = stats.norm.rvs(loc=0, scale=1, size=num_data_points) 

    if plotting:
        xmin = stats.norm.ppf(0.01) # at 1% probability
        xmax = stats.norm.ppf(0.99)# at 90% probability
        x = np.linspace(xmin, xmax, num_data_points)
        y1 = stats.norm.pdf(x)
        y2 = stats.norm.cdf(x)

        fig, ax = plt.subplots(1, 1)
        ax.plot(x, y1,'k-', lw=3, label='norm pdf')
        ax.plot(x, y2, 'r.', lw=4, label='norm cdf')
        ax.legend(loc='best', frameon=False)
        plt.show()
    return samples

