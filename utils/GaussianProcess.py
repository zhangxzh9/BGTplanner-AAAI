
import math
import numpy as np 
import random

from scipy.spatial.distance import cdist
class GaussianProcessRegressor(object):
    def __init__(self):
        # self.args = args
        self.epsilon = 0.001
        self.c_1 = 0.8
        self.c_2 = 4
        self.l = 0.2
        self.kernel_arg="se"
        self.sigma = 0.1


    def fit(self, t, x_train, x_predict):
        t = x_train.shape[0]
        K = self.kernel(x_train, x_train)
        D = np.zeros_like(K)
        for i in range(t):
            for j in range(t):
                D[i][j] = (1 - self.epsilon) ** (abs(i - j) / 2)
        K = K * D
        K_star = self.kernel(x_train, x_predict)
        d = np.zeros_like(K_star)
        for i in range(t):
            d[i] = (1 - self.epsilon) ** ((t + 1 - (i + 1)) / 2)
        K_star = K_star * d
        K_2stars = self.kernel(x_predict, x_predict)
        return K, K_star, K_2stars

    def predict(self, K, K_star, K_2stars, y_train):
        noise = np.ones_like(K) * (self.sigma ** 2)
        K_inv = np.linalg.inv((K + noise))
        mean = (K_star.T).dot(K_inv).dot(y_train)
        cov = K_2stars - (K_star.T).dot(K_inv).dot(K_star)
        std = np.sqrt(cov.diagonal())
        return mean, std

    def kernel(self, x_1, x_2):
        if self.kernel_arg == 'se':
            result = self.squared_exponential_kernel(x_1, x_2)
        return result

    def squared_exponential_kernel(self, x_1, x_2):
        temp = cdist(x_1, x_2)
        # temp = np.linalg.norm(x_1 - x_2)
        result = np.exp(-0.5 * ((temp / self.l) ** 2))
        return result

    def sample_func_from_gp(self, mean, cov, func_num):
        y = np.random.multivariate_normal(
            mean=mean, cov=cov,
            size=func_num)
        return y
    
    def cal_beta(self,epoch_index):
        beta = self.c_1 * math.log(self.c_2 * (epoch_index + 1))
        return beta