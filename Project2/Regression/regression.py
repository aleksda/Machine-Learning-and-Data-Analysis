import numpy as np

class Manual_regression:
    def __init__(self, scaler_obj):
        """
        Takes in an object split_scale. That's to insure that the user first
        scales the data before doing any analysis. The scaled data is saved by 
        and accessed through the self.scaler instance variable.

        Parameters
        ----------
        scaler_obj : A split_scale instance.

        Returns
        -------
        None.

        """
        self.scaler = scaler_obj
        self.lmb = 0
        
    def OLS(self, X, Z):
        X_T = X.T
        return np.linalg.pinv(X_T @ X) @ X_T @ Z
    
    def Ridge(self, X, Z):
        lmb = self.lmb
        p = X.shape[1]
        X_T = X.T
        I = np.eye(p, p)
        return np.linalg.pinv(X_T @ X + lmb*I) @ X_T @ Z
    
    def fit(self, X, Z, method= "OLS"):
        if method in ["ols", "OLS", "Ols"]:
            return self.OLS(X, Z)
        elif method in ["ridge", "Ridge", "RIDGE"]:
            return self.Ridge(X, Z)
        elif method in ["lasso", "Lasso", "LASSO"]:
            print("Numpy scaler doesn't allow Lasso, choose Scikit!")
            
    def predict(self, X, beta):
        # print("Manual regression predict")
        return X @ beta
            
class Scikit_regression:
    def __init__(self, scaler_obj):
        self.scaler = scaler_obj
        self.lmb = 0
        
    def OLS(self, X, Z):
        from sklearn.linear_model import LinearRegression 
        linreg = LinearRegression()
        return linreg.fit(X, Z)
        
    def Ridge(self, X, Z):
        from sklearn.linear_model import Ridge
        lmb = self.lmb
        ridge = Ridge(alpha= lmb)
        return ridge.fit(X, Z)
        
    def Lasso(self, X, Z):
        from sklearn.linear_model import Lasso
        lmb = self.lmb
        # lasso = Lasso(alpha= lmb, tol= 1e-3, max_iter= 1e6)
        lasso = Lasso(alpha= lmb, tol= 0.1, max_iter= 1e6)
        # lasso = Lasso(lmb)
        # RegLasso = linear_model.Lasso(lmb)
        return lasso.fit(X, Z)
        
    def fit(self, X, Z, method= "OLS"):
        if method in ["ols", "OLS", "Ols"]:
            return self.OLS(X, Z)
        elif method in ["ridge", "Ridge", "RIDGE"]:
            return self.Ridge(X, Z)
        elif method in ["lasso", "Lasso", "LASSO"]:
            return self.Lasso(X, Z)
        
    def predict(self, X, beta):
        return beta.predict(X)
    
class Logistic_regression:
    def __inint__(self):
        self.lmb = 0 # l2 pa
        
    def Sigmoid(self, t):
        # print("#! wow")
        return 1/(1 + np.exp(-t))

    def predict(self, inputs, weights):
        # print("#2 wow")
        # print("dim weights: ", weights.shape, " dim inputs: ", inputs.shape)
        return self.Sigmoid(np.matmul(inputs, weights))