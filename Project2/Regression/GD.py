import error as err

# =============================================================================
# Python Libraries
# =============================================================================
import numpy as np
from sys import exit
from sklearn.metrics import mean_squared_error



class Gradient_descent:
    def __init__(self, X, Z, reg_obj):
        print("Class Gradient_descent initialized")
        
        self.reg = reg_obj # regression
        
        self.N = X.shape[0]
        self.p = X.shape[1]
        print("N: ", self.N, " p: " , self.p)   
        
        # self.epsilon = 1e-08
        
        if len(Z.shape) == 1:
            Z = Z[:,np.newaxis]
        
        self.X = X
        self.Z = Z
        print("X:", X.shape, "Z:", Z.shape)
        
        self.K = Z.shape[1] # Number of classes/categories/dimensions in the response 
        self.lmb = 0
        self.gamma = 0.2
        self.start_beta = np.random.randn(self.p, self.K) 
        
        # Hessian matrix (should probably remove)
        N= self.N; X_T = X.T;
        
        H = (2/N) * X_T @ X
        EigValues, EigVectors = np.linalg.eig(H)
        
        learning_rate = 1.0/np.max(EigValues)
        if learning_rate > 1e-04:
            self.eta = learning_rate
        else:
            self.eta = 1e-04
        # print("eta: ", learning_rate)
    
    def Regression_gradient(self, X, Z, beta, method= "OLS"):
        X_T = X.T; N = X.shape[0]
        
        if method in ["ols", "OLS", "Ols"]:
            gradient = (2.0/N)*X_T @ (X @ beta - Z)
        elif method in ["ridge", "Ridge", "RIDGE"]:
            lmb = self.lmb
            gradient = (2.0/N)*X_T @ (X @ beta - Z) + 2*lmb*beta
        
        return gradient
    
    def Sigmoid(self, t):
        return 1/(1 + np.exp(-t))
    
    def logistic_prediction(self, weights, inputs):
        # print("dim weights: ", weights.shape, " dim inputs: ", inputs.shape)
        return self.Sigmoid(np.matmul(inputs, weights))
    
    def cross_entropy_grad(self, X, Z, beta, method= "Logistic Regression"):
        X_T = X.T
        # beta is the same as weights
        p = self.logistic_prediction(beta, X)
        if not method == "L2-regularization":
            return X_T @ (p - Z)
        else:
            lmb = self.lmb
            return X_T @ (p - Z) + 2*lmb*beta
    
    def Convergence_test(self, method):
        if method in ["ols", "OLS", "Ols", "ridge", "Ridge", "RIDGE"]:
            # diff = abs(error_cache[epoch+1] - error_cache[epoch])
            self.epsilon = 1e-08
            return lambda var, idx : np.abs(var[idx+1] - var[idx])
        else:
            self.epsilon = 0.08
            return lambda var, idx : np.abs(1 - var[idx+1])
            # self.epsilon = 1e-08
            # return lambda var, idx : np.abs(np.abs(1 - var[idx + 1]) - np.abs(1 - var[idx]))
            
        
    
    
    def Gradient(self, method= "OLS"):
        if method in ["ols", "OLS", "Ols", "ridge", "Ridge", "RIDGE"]:
            diff_func = self.Convergence_test(method)
            return self.Regression_gradient, mean_squared_error, diff_func
        else:
            error = err.Error(self.Z)
            accuracy = error.Accuracy
            diff_func = self.Convergence_test(method)
            return self.cross_entropy_grad, accuracy, diff_func
    
    def Simple_GD(self, initial_guess, method= "OLS", Niterations= 300,\
                  learning_rate= None, plotting = False):
        beta = initial_guess.copy()
        X = self.X; Z = self.Z;
        
        if learning_rate == None:
            eta = self.eta
        else:
            eta = learning_rate
        
        
        print(f"\nSimple GD {method}")
        print(f"Learning rate: {eta}")
        
        error_cache = []
        
        n = 10000 # Default number of iterations not adjustable by user
        
        if method in ["ridge", "Ridge", "RIDGE"] or method == "L2-regularization":
            print("lmb: ", self.lmb)
        
        gradient_func, error_func, diff_func = self.Gradient(method= method)
        
        if not plotting:
            for i in range(n):
                gradient = gradient_func(X, Z, beta, method= method)
                beta -= eta*gradient
                
        elif plotting:
            print("Plotting preset initiated")
            n = 0
                
            
        
        if Niterations > n:
            # print("I'm bigger than you thought!")
            Z_tilde = self.reg.predict(X, beta)
            error = error_func(Z, Z_tilde)
            error_cache.append(error)
            
            for i in range(Niterations - n):
                gradient = gradient_func(X, Z, beta, method= method)
                beta -= eta*gradient
                
                # Convergence test
                Z_tilde = self.reg.predict(X, beta)
                error = error_func(Z, Z_tilde)
                error_cache.append(error)
                
                diff = diff_func(error_cache, i)
                
                if diff < self.epsilon:
                    break
                
                # if np.abs(gradient).max() < 1e-08:
                #     break
                
            print(f"Stopped GD {method} at iteration i: {i + n} \n")
        
        return beta, error_cache, i+n
    
    def Stochastic_GD(self, initial_guess, method= "OSL", batch_size= 5,\
                      epochs= 30, learning_rate= None, plotting= False):
        beta = initial_guess.copy()
        X = self.X; Z = self.Z
        N = self.N; p = self.p; K = self.K
        # print("N: ", N, "p: ", p)
        
        if learning_rate == None:
            eta = 1e-04
        else:
            eta = learning_rate
        
        
        n = 30 # Default number of epochs not adjustable by user
        gamma = self.gamma
        running_avg = np.zeros((p, K))
        
        print(f"\nStochastic GD {method}")
        print(f"Learning rate: {eta}")
        print(f"Gamma: {gamma}")
        
        error_cache = []
        
        M = batch_size
        m = int(N/M) # number of mini-batches
        
        print(f"Data points: {N}, Batch size: {M}, number of min-batches: {m}")
        
        
        
        if method in ["ridge", "Ridge", "RIDGE"] or method == "L2-regularization":
            print("lmb: ", self.lmb)
        
        if N%M != 0:
            print(f"Size of mini-batches M = {M} is not divisble by N = {N}")
            exit()
            
        gradient_func, error_func, diff_func = self.Gradient(method= method)
        
        indices = np.arange(N)
        if not plotting:
            for epoch in range(n):    
                rnd_indices = np.copy(indices)
                np.random.shuffle(rnd_indices)
                for i in range(m):
                    # k = np.random.randint(m)
                    # batch = rnd_indices[k*M: k*M + M]
                    batch = rnd_indices[i*M: i*M + M]
                    
                    X_ = X[batch]
                    Z_ = Z[batch]
                    
                    # print(batch.shape, X_.shape, Z_.shape)
                    
                    # gradient = gradient_func(X_, Z_, beta, method= method)
                    # beta -= eta*gradient
                    
                    # momentum
                    gradient = gradient_func(X_, Z_, beta, method= method)
                    running_avg = gamma*running_avg + eta*gradient
                    beta -= running_avg
                    
        elif plotting:
            print("Plotting preset initiated")
            n = 0
        
        if epochs > n:
            # print("I'm bigger than you thought!")
            Z_tilde = self.reg.predict(X, beta)
            error = error_func(Z, Z_tilde)
            error_cache.append(error)
            for epoch in range(epochs - n):  
                rnd_indices = np.copy(indices)
                np.random.shuffle(rnd_indices)
                for i in range(m):
                    # k = np.random.randint(m)
                    # batch = rnd_indices[k*M: k*M + M]
                    batch = rnd_indices[i*M: i*M + M]
                    
                    X_ = X[batch]
                    Z_ = Z[batch]
                    
                    # gradient = gradient_func(X_, Z_, beta, method= method)
                    # beta -= eta*gradient
                    
                    # momentum
                    gradient = gradient_func(X_, Z_, beta, method= method)
                    running_avg = gamma*running_avg + eta*gradient
                    beta -= running_avg
                    
                    
                # Convergence test
                Z_tilde = self.reg.predict(X, beta)
                error = error_func(Z, Z_tilde)
                error_cache.append(error)
                
                diff = diff_func(error_cache, epoch)
                
                if diff < self.epsilon:
                    break
            
            print(f"Stopped SGD {method} at epoch: {n + epoch} \n")
        return beta, error_cache, n+epoch