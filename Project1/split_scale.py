import numpy as np
from sklearn.model_selection import train_test_split


class Numpy_split_scale:
    """
    Creates a design matrix using NumPy. Splits the data into a training-set 
    and a testing-set, using Scikit. Then scales it using NumPy.
    It also scales the data without splitting for cross-validation.
    
    """
    def __init__(self, x, y, Z, deg= 3, test_size= 0.2):
        print("Class Numpy_split_scale initializer")
        self.x = x; self.y = y; self.Z = Z; self.deg = deg
        # print("Deg:", deg)
        
        self.X = self.create_design_matrix(x, y, deg)
        self.X_scaled = self.numpy_scaler(self.X)
        
        X_train, X_test, Z_train, Z_test = train_test_split(self.X, self.Z, test_size= test_size)
        
        self.X_train_scaled = self.numpy_scaler(X_train)
        self.X_test_scaled = self.numpy_scaler(X_test)
        
        # self.X_train_scaled = X_train
        # self.X_test_scaled = X_test
        
        self.Z_train = Z_train; self.Z_test = Z_test
        # self.Z_train = self.numpy_scaler(Z_train); self.Z_test = self.numpy_scaler(Z_test)
        # print(self.x.shape, self.y.shape, self.Z.shape, self.X.shape, self.X_scaled.shape,\
        #       self.X_train_scaled.shape, self.X_test_scaled.shape, self.Z_train.shape, self.Z_test.shape)
        
    def create_design_matrix(self, x, y, deg):
        """
        Creates a design matrix with columns:
        [1  x  y  x^2  y^2  xy  x^3  y^3  x^2y ...]
        """
        # print("Numpy_split_scale create_design_matrix")
        if len(x.shape) > 1:
            x = np.ravel(x)
            y = np.ravel(y)

        N = len(x)
        p = int((deg + 1)*(deg + 2)/2)
        X = np.ones((N,p))
        
        for i in range(1, deg + 1):
            q = int((i)*(i+1)/2)
            for k in range(i+1):
                X[:,q+k] = x**(i-k) * y**k
        
        return X
    
    def numpy_scaler(self, X):
        """
        Scaling a polynolmial matrix using NumPy
        
        Parameters
        ----------
        X : Matrix to be scaled

        Returns
        -------
        X : Scaled matrix

        """
        X_ = X[:,1:] # Ommitting the intercept
        
        # print("----------------------Numpy--------------------------------")
        mean = np.mean(X_, axis=0); std = np.std(X_, axis=0); # var = np.var(X_, axis=0)
        X_scaled = (X_ - mean)/std
        # print("Dim X: ", X.shape, " Dim X_:", X_scaled.shape)
        # print("np.mean: ", mean, " mean.shape: ", mean.shape, "np.var: ", var, " var.shape: ", var.shape )
        # print("X_np_scaled[6]: ", X_scaled[6])
        # print("------------------------------------------------------------\n")
        
        X = np.insert( X_scaled, 0, X[:,0], axis= 1 ) # Reconstructing X after ommiting the intercept
        
        return X
    
class Scikit_split_scale:
    """
    Creates a design matrix using Scikit. Splits the data into a training-set 
    and a testing-set then scales it using Scikit. It also scales the data 
    without splitting for cross-validation.
    
    """
    def __init__(self, x, y, Z, deg= 3, test_size= 0.2):
        print("Class Scikit_split_scale initializer")
        self.x = x; self.y = y; self.Z = Z; self.deg = deg
        # print("Deg:", deg)
        
        self.X = self.scikit_design_matrix(x, y, deg)
        self.X_scaled = self.scikit_scaler(self.X) #Scales X for Cross-validation later on
        
        X_train, X_test, Z_train, Z_test = train_test_split(self.X, self.Z, test_size= test_size)
        
        self.X_train_scaled = self.scikit_scaler(X_train) #Under scaling the training data behaves as expected
        self.X_test_scaled = self.scikit_scaler(X_test) #The test MSE is scattered everywhere and is unpredictable
        
        # self.X_train_scaled = X_train #Use this line so you want be scaling the data without doing much changes to the code.
        # self.X_test_scaled = X_test #Looks like it tends to overfit with some discrepancies, but that doesn't make any sense.
        
        self.Z_train = Z_train; self.Z_test = Z_test
        # self.Z_train = self.scikit_scaler(Z_train); self.Z_test = self.scikit_scaler(Z_test) #Scales the response data
        # print(self.x.shape, self.y.shape, self.Z.shape, self.X.shape, self.X_scaled.shape,\
        #       self.X_train_scaled.shape, self.X_test_scaled.shape, self.Z_train.shape, self.Z_test.shape)
    
    def scikit_design_matrix(self, x, y, deg):
        """
        Creates a design matrix with columns, using scikit:
        [1  x  y  x^2  y^2  xy  x^3  y^3  x^2y ...]
        """
        from sklearn.preprocessing import PolynomialFeatures
        
        if len(x.shape) > 1:
            x = np.ravel(x)
            y = np.ravel(y)
        
        # X = np.vstack((x,y)).T
        X = np.stack((x,y), axis = 1)
        # X = np.stack((x,y), axis = -1)
        poly = PolynomialFeatures(deg)
        X = poly.fit_transform(X)
        
        return X
    
    def scikit_scaler(self, X):
        """
        Scaling a polynolmial matrix using StandardScaler from Scikit-learn
        
        Parameters
        ----------
        X : Matrix to be scaled

        Returns
        -------
        X : Scaled matrix

        """
        from sklearn.preprocessing import StandardScaler
        
        X_ = X[:,1:] # Ommitting the intercept
        
        # print("---------------------Scikit---------------------------------")
        scaler = StandardScaler()
        # scaler.fit(X_)
        # X_scaled = scaler.transform(X_)
        # print("X.shape: ", X[:,1:].shape, " mean: ", scaler.mean_, " shape: ", scaler.mean_.shape, "var:", scaler.var_ )
        # print("X_scaled[6]: ", X_scaled[6])
        X_scaled = scaler.fit_transform(X_)
        # print("X_scaled[6] back to back: ", X_scaled[6])
        # print("------------------------------------------------------------\n")
        
        X = np.insert( X_scaled, 0, X[:,0], axis= 1 ) #Reconstructing X after ommiting the intercept
        
        return X