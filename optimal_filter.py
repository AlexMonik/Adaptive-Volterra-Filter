import pandas as pd
import numpy as np
import datetime as dt


class OptimalFilter:
    def __init__(self, typ):
        """
        :param typ: 1 - linear filter; 2 - nonlinear filter
        """
        self.weights = None
        self.typ = typ

    def fit(self, x, y):
        """
        Calculate optimal weights of filter.
        :param x: inputs (pandas.DataFrame or numpy.array with shape of M x N, M - number of realizations,
                                                                               N - number of signal's samples)
        :param y: outputs (M x 1)
        :return: optimal weights (numpy.array; if typ=1 - linear weights, if typ=2 - linear and nonlinear)
        """
        if self.typ == 2:
            x = OptimalFilter.trans(x)
        elif self.typ == 1:
            x = np.array(x)
        else:
            raise NameError('type of filter should be 1 (linear) or 2 (nonlinear) ')
        y = np.array(y)
        r = OptimalFilter.corr(x, x)
        p = OptimalFilter.corr(x, y)
        self.weights = np.linalg.inv(r) @ p

    def predict(self, x):
        """
        Predict output by input vector.
        :param x: input vector (pandas.DataFrame or numpy.array with shape of M x N, M - number of realizations,
                                                                                     N - number of signal's samples)
        :return: output vector
        """
        if self.weights is None:
            raise NameError('The weights of filter are undefined. Call method .fit(inputs, outputs) first.')
        if self.typ == 2:
            x = OptimalFilter.trans(x)
        elif self.typ == 1:
            x = np.array(x)
        else:
            raise NameError('type of filter should be 1 (linear) or 2 (nonlinear) ')
        predicted = np.zeros([x.shape[0], 1])
        for i in range(x.shape[0]):
            predicted[i, 0] = self.weights.T @ x[i, :].reshape([x.shape[1], 1])
        return predicted.reshape(x.shape[0])

    @staticmethod
    def corr(x, y):
        """
        correlation matrix
        """
        r = np.zeros([x.shape[0], x.shape[1], y.shape[1]])
        for i in range(x.shape[0]):
            a = x[i, :].reshape([x.shape[1], 1])
            b = y[i, :].reshape([1, y.shape[1]])
            r[i, :, :] = a @ b
        return r.mean(axis=0)

    @staticmethod
    def trans(x):
        """
        Transform input signal vector into nonlinear form (second order)
        TODO add up-to p-th order
        :param x: linear form of input signal vector
        :return: nonlinear form of input signal vector ([linear part, nonlinear part])
        """
        x = np.array(x)
        n = 0
        for i in range(x.shape[1]):
            n += (i+1)
        ux = np.zeros([x.shape[0], x.shape[1]+n])
        for k in range(x.shape[0]):
            j1 = x.shape[1]
            for i in range(x.shape[1]):
                ux[k, i] = x[k, i]
                for j in range(i, x.shape[1]):
                    ux[k, j1] = x[k, i]*x[k, j]
                    j1 += 1
        return ux
