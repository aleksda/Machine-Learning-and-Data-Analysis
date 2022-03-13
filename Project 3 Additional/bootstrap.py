import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

class Bootstrap:
    def __init__(self, x, y, t, seed, nr_of_boots = 100, scaler=StandardScaler, scale_target=True):

        self.x = np.ravel(x).reshape(np.size(x), 1)
        self.y = np.ravel(y).reshape(np.size(y), 1)
        self.t = np.ravel(t)

        self.X = np.hstack((self.x, self.y))

        self.nr_of_boots = nr_of_boots

        self.X_train, self.X_test, self.t_train, self.t_test = train_test_split(self.X, self.t, test_size = 0.2, random_state = seed)

        self.__scale()

    def simulate(self, model):

        X_train, X_test, t_train, t_test = self.X_train, self.X_test, self.t_train, self.t_test
        t_train = t_train.ravel(); t_test = t_test.ravel()

        t_tilde = np.zeros([t_train.shape[0], self.nr_of_boots])
        t_pred = np.zeros([t_test.shape[0], self.nr_of_boots])

        for boot in range(self.nr_of_boots):
            X_bt, t_bt = resample(X_train, t_train)

            model.fit(X_bt, t_bt)
            t_tilde[:, boot] = model.predict(X_train)
            t_pred[:, boot]  = model.predict(X_test)

        return t_tilde, t_pred

    def decompose(self, t_pred):

        t_test = self.t_test

        mse = np.mean(np.mean((self.t_test - t_pred)**2, axis = 1, keepdims = True))
        bias = np.mean((t_test - np.mean(t_pred, axis = 1, keepdims = True))**2)
        var = np.mean((t_pred - np.mean(t_pred, axis = 1, keepdims = True))**2)

        return mse, bias, var

    def __scale(self):

        scale_X = StandardScaler()
        scale_X.fit(self.X_train)
        self.X_train = scale_X.transform(self.X_train)
        self.X_test = scale_X.transform(self.X_test)

        scale_t = StandardScaler()
        scale_t.fit(self.t_train.reshape(-1, 1))
        self.t_train = scale_t.transform(self.t_train.reshape(-1, 1))
        self.t_test = scale_t.transform(self.t_test.reshape(-1, 1))