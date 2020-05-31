import numpy as np


class LMSFilter:
    def __init__(self, typ, mu1, mu2=None, complx=False):
        """
        :param typ: 1 - linear filter; 2 - nonlinear filter
        :param mu1: step-size for the first order weights
        :param mu2: step-size for the second order weights (required only if typ=2)
        :param complx: is input data complex-valued
        """
        self.weights = None
        self.typ = typ
        self.mu1 = mu1
        self.mu2 = mu2
        self.complx = complx

    def train(self, x, y):
        """
        Calculate optimal weights of filter.
        :param x: inputs (pandas.DataFrame or numpy.array with shape of M x N, M - number of realizations,
                                                                               N - number of signal's samples)
        :param y: outputs (M x 1)
        :return: error and optimal weights (numpy.array; if typ=1 - linear weights, if typ=2 - linear and nonlinear)
        """
        Mu1 = [self.mu1 for i in range(x.shape[1])]
        y = np.array(y)
        if self.typ == 2:
            if self.mu2 is None:
                raise NameError('step-size mu2 must be set')
            N1 = x.shape[1]
            x = LMSFilter.trans(x)
            Mu2 = [self.mu2 for i in range(x.shape[1]-N1)]
            M = np.diagflat(Mu1+Mu2)
        elif self.typ == 1:
            x = np.array(x)
            M = np.diagflat(Mu1)
        else:
            raise NameError('type of filter should be 1 (linear) or 2 (nonlinear) ')
        self.weights = np.zeros([x.shape[0]+1, x.shape[1]])
        filter_output = np.zeros([x.shape[0], 1])
        errors = np.zeros([y.shape[0], 1])
        if self.complx:
            for i in range(x.shape[0]):
                filter_output[i, 0] = self.weights[i, :] @ x[i, :].T
                errors[i, 0] = y[i] - filter_output[i]
                self.weights[i+1, :] = self.weights[i, :] + x[i, :] @ M * errors[i, 0]
        else:
            for i in range(x.shape[0]):
                filter_output[i, 0] = self.weights[i, :] @ x[i, :].T
                errors[i, 0] = y[i] - filter_output[i]
                self.weights[i+1, :] = self.weights[i, :] + 2*x[i, :] @ M * errors[i, 0]
        return [errors, self.weights]

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
            x = LMSFilter.trans(x)
        elif self.typ == 1:
            x = np.array(x)
        else:
            raise NameError('type of filter should be 1 (linear) or 2 (nonlinear) ')
        predicted = np.zeros([x.shape[0], 1])
        for i in range(x.shape[0]):
            predicted[i, 0] = self.weights[-1, :] @ x[i, :].T
        return predicted.reshape(x.shape[0])

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
