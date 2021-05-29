import numpy as np
import statsmodels.api as sm
#statsmodels-0.12.2 required
from statsmodels.tsa.arima_process import arma_generate_sample
from solution import solution

np.random.seed(0)
arparams, maparams = np.array([0.6, -0.13, -0.38, 0.53]), np.array([])
ar, ma = np.r_[1, -arparams], np.r_[1, maparams]

y_sigma = arma_generate_sample(ar, ma, nsample=1000, distrvs=np.random.normal, scale=1)

print(solution(y_sigma, len(arparams)))
#output: array([ 0.59571561, -0.1014831 , -0.32694882,  0.47210408])
