class Filter:
    def __init__(self, filter_type):
        self.weights = None
        self.filter_type = filter_type
        if self.filter_type != 2 and self.filter_type != 1:
            raise NameError('Type of filter should be 1 (linear) or 2 (nonlinear)')

    def fit(self, x, y):
        """
        calculate optimal weights
        :param x: [list or numpy.nd array] input signal; 1st dimension - signal's realization (N)
                                                         2nd dimension - signal's samples (M)
        :param y: [list or numpy.nd array] output signal; shape must be (N,) or (N,1)
        """
        x = Filter.prepare_data(x)
        y = Filter.prepare_data(y)
        if x.shape[0] != y.shape[0]:
            raise NameError('The number of input\'s and output\'s realizations is not the same')
        if self.filter_type == 2:
            x = Filter.trans(x)
        r = Filter.corr(x, x)
        p = Filter.corr(x, y)
        self.weights = np.linalg.pinv(r) @ p
        del r, p, x, y

    def predict(self, x):
        if self.weights is None:
            raise NameError('The weights of the filter are undefined. Call method .fit(input, output) first.')
        x = Filter.prepare_data(x)
        if x.shape == (max(x.shape), 1):  # костыли
            x = x.T
        if self.filter_type == 2:
            x = Filter.trans(x)
        predicted = []
        for i in range(x.shape[0]):
            rez = self.weights.T @ x[i, :]
            predicted.append(rez[0])
        return predicted

    @staticmethod
    def prepare_data(x):
        if type(x) is not np.ndarray and type(x) is not list:
            raise NameError('The type of signal must be either numpy.ndarray or list')
        if type(x) is list:
            x = np.array(x)
        if len(x.shape) > 2:
            raise NameError('The number of dimensions of signal can not be greater then 2')
        if len(x.shape) == 1:
            x = x.reshape((x.shape[0], 1))
        return x

    @staticmethod
    def corr(x, y):
        """
        correlation matrix
        """
        r = np.zeros([x.shape[0], x.shape[1], y.shape[1]])
        for i in range(x.shape[0]):
            a = x[i, :].reshape((x.shape[1], 1))
            b = y[i, :].reshape((1, y.shape[1]))
            r[i, :, :] = a @ b
        return r.mean(axis=0)

    @staticmethod
    def trans(x):
        """
        Transform input signal vector into nonlinear form (second order)
        TODO add up-to p-th order
        :param x: linear form of input signal vector
        :return: nonlinear form of input signal vector
        """
        n = 0
        for i in range(x.shape[1]):
            n += i + 1
        ux = np.zeros((x.shape[0], x.shape[1] + n))
        for k in range(x.shape[0]):
            j1 = x.shape[1]
            for i in range(x.shape[1]):
                ux[k, i] = x[k, i]
                for j in range(i, x.shape[1]):
                    ux[k, j1] = x[k, i] * x[k, j]
                    j1 += 1
        return ux
    
    @staticmethod
    def make_s(a, M):
        """
        rearrange time-series to the array of filter's inputs
        TODO now its work only with list; make it work with numpy.ndarray
        :param a: origin_list
        :param M: filter's tap
        :return: new_x
        """
        x = []
        y = []
        for r in range(len(a) - M):
            row = []
            for c in range(M):
                row.append(a[r + c])
            x.append(row)
            y.append(a[r + M])
        return x, y
