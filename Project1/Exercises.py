import frankefunction as fr
import error as err
import  split_scale as split
import regression as reg
import resampling as res
import output as out

# =============================================================================
# Python Libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

def train_test_error(scaler_obj, regression_class, method= "OLS", lmb = 0, printToScreen = False, defaultError = "Scikit"):
    print("Method: ", method, "lmb: ",  lmb, "print 0/1: ", printToScreen, "Error module: ",  defaultError)
    if regression_class == "Scikit":
        Scikit_REG = reg.Scikit_regression(scaler_obj)
        Scikit_REG.lmb = lmb
        beta = Scikit_REG.fit(scaler_obj.X_train_scaled, scaler_obj.Z_train, method= method)
    else:
        Manual_REG = reg.Manual_regression(scaler_obj)
        Manual_REG.lmb = lmb
        beta = Manual_REG.fit(scaler_obj.X_train_scaled, scaler_obj.Z_train, method= method)
        
    
    error_printout = out.PrintError(scaler_obj)
    if defaultError == "Scikit":
        error_printout.ScikitError(beta, printToScreen= printToScreen)
    else:
        error_printout.ManualError(beta, printToScreen= printToScreen)
        
    return error_printout.ErrorDict

def confidence_interval(scaler_obj, regression_class, var= 0, method= "OLS", lmb = 0, errorbar= False):
    print("var1: ", var)
    if var == None:
        var = np.var(scaler_obj.X[:,1:])
    print("var2: ", var)
    X = scaler_obj.X_train_scaled; X_T = X.T
    var_beta_numpy = var*np.diag(np.linalg.pinv(X_T @ X))
    std_beta = np.sqrt(var_beta_numpy)
    CI = 1.96*std_beta
    
    if regression_class == "Scikit":
        Scikit_REG = reg.Scikit_regression(scaler_obj)
        Scikit_REG.lmb = lmb
        beta = Scikit_REG.fit(scaler_obj.X_train_scaled, scaler_obj.Z_train, method= method)
        beta_array = np.copy(beta.coef_) 
        beta_array[0] += beta.intercept_
        beta = beta_array
        x = np.arange(len(beta))
        plt.figure()
        plt.title(f"Scikit Regression of a polynomial degree $p= {scaler_obj.deg}$")
    else:
        Manual_REG = reg.Manual_regression(scaler_obj)
        Manual_REG.lmb = lmb
        beta = Manual_REG.fit(scaler_obj.X_train_scaled, scaler_obj.Z_train, method= method)
        x = np.arange(len(beta))
        plt.figure()
        plt.title(f"Manual Regression of a polynomial degree $p= {scaler_obj.deg}$")
    
    
    # plt.style.use("seaborn-whitegrid")
    plt.style.use("seaborn-darkgrid")
    plt.xticks(x)
    
    plt.errorbar(x, beta, CI, fmt=".", capsize= 3, label=r'$\beta_j \pm 1.96 \sigma$')
    plt.legend(edgecolor= "black")
    plt.xlabel(r'index $j$')
    plt.ylabel(r'$\beta_j$')
    plt.show()
    
    if errorbar:
        plt.figure()
        plt.title(r"CI estimate for each $\beta$")
        plt.bar(x, CI)
        plt.xlabel(r'index $j$')
        plt.ylabel(r'CI($\beta_j$)')
        plt.legend()
        plt.show()
        
def train_vs_test_MSE(xx, yy, z_flat, regression_class, maxdegree= 10, method= "OLS", lmb= 0, cv= False):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score,\
        mean_squared_log_error, mean_absolute_error
    
    maxdegree = maxdegree + 1; start = 1; degrees = np.arange(start, maxdegree)
    # Error_list = ["R2 Score", "MSE", "MAE"]
    # Error_list = ["Error", "Bias", "Variance"]
    # degree_analysis = np.zeros((maxdegree - start, 3, 2)) # [degree, Error[0]/Bias[1]/Variance[2], Predict[0]]
    n_bootstraps = 100
    
    trainerror = np.zeros(maxdegree - start)
    testerror = np.zeros(maxdegree - start)
    CV_MSE = np.zeros(maxdegree - start)
    time_CV = np.zeros(maxdegree - start)
    time_Boot = np.zeros(maxdegree - start)
    
    if regression_class == "Scikit":
        for deg in degrees:
            print(f"----------------DEGREE: {deg}-----------------")
            scale_split = split.Scikit_split_scale(xx, yy, z_flat, deg= deg)
            
            if cv:
                t0 = time.time()
                CV_MSE[deg - start] = cross_validation(scale_split, method = method, k = 10, lmb= lmb)
                time_CV[deg - start] = time.time() - t0
            
            
            #Bootstrap
            t1 = time.time()
            
            for boot in range(n_bootstraps):
                X_train, X_test, Z_train, Z_test = train_test_split(scale_split.X, scale_split.Z, test_size= 0.2)
                X_train_scaled = scale_split.scikit_scaler(X_train); X_test_scaled = scale_split.scikit_scaler(X_test)
                
                Scikit_REG = reg.Scikit_regression(scale_split)
                Scikit_REG.lmb = lmb
                
                beta = Scikit_REG.fit(X_train_scaled, Z_train, method= method)
                Z_tilde = beta.predict(X_train_scaled)
                Z_pred  = beta.predict(X_test_scaled)
                
                trainerror[deg - start] += mean_squared_error(Z_train, Z_tilde) 
                testerror[deg - start] += mean_squared_error(Z_test, Z_pred)
            
            time_Boot[deg - start] = time.time() - t1
        
        plt.figure()
        plt.title("Scikit Regression")
    
    else:
        for deg in degrees:
            print(f"----------------DEGREE: {deg}-----------------")
            scale_split = split.Numpy_split_scale(xx, yy, z_flat, deg= deg)
            
            if cv:
                t0 = time.time()
                CV_MSE[deg - start] = cross_validation(scale_split, method = method, k = 10, lmb= lmb)
                time_CV[deg - start] = time.time() - t0
            
            #Bootstrap
            t1 = time.time()
            for boot in range(n_bootstraps):
                X_train, X_test, Z_train, Z_test = train_test_split(scale_split.X, scale_split.Z, test_size= 0.2)
                X_train_scaled = scale_split.numpy_scaler(X_train); X_test_scaled = scale_split.numpy_scaler(X_test)
                
                Manual_REG = reg.Manual_regression(scale_split)
                Manual_REG.lmb = lmb
                
                beta = Manual_REG.fit(X_train_scaled, Z_train, method= method)
                Z_tilde   = X_train_scaled @ beta
                Z_pred    = X_test_scaled @ beta
                
                trainerror[deg - start] += mean_squared_error(Z_train, Z_tilde) 
                testerror[deg - start] += mean_squared_error(Z_test, Z_pred)
            
            time_Boot[deg - start] = time.time() - t1
        plt.figure()
        plt.title("Manual Regression")
    
    trainerror /= n_bootstraps
    testerror /= n_bootstraps
    
    for deg in degrees:
        print(f"----------------DEGREE: {deg}-----------------")
        print("Degree of polynomial: %3d" %deg)
        print("Mean squared error on training data: %.8f" % trainerror[deg - start])
        print("Mean squared error on test data: %.8f" % testerror[deg - start])
    
    if cv == False:
        plt.plot(degrees, np.log10(trainerror), "o-", label='Training Error')
        plt.plot(degrees, np.log10(testerror), "o-", label='Test Error')
    else:
        plt.plot(degrees, np.log10(testerror), "o-", label='Test Error')
        plt.plot(degrees, np.log10(CV_MSE), "o-", label= "CV Test Error")
    plt.xlabel('Polynomial degree')
    plt.ylabel('log10[MSE]')
    plt.legend()
    plt.show()
    # save_fig(train_vs_test_MSE.__name__ + regression_class + method + "CV" )
    # save_fig(train_vs_test_MSE.__name__ + regression_class + "terrainCV" + str(int(np.log10(lmb))) )
    
    if cv == "21321":
        # print(time_Boot, time_CV)
        plt.figure()
        plt.title("Measured time: Boot vs. Cross-validation")
        plt.plot(degrees, time_Boot, label= "Boot time")
        plt.plot(degrees, time_CV, label= "CV time")
        plt.xlabel('Polynomial degree')
        plt.ylabel("t [s]")
        plt.legend()
        plt.show()
        # save_fig(train_vs_test_MSE.__name__ + regression_class + "terrainCVTime" + str(int(np.log10(lmb))))
    
def bias_variance_tradeoff(xx, yy, z_flat, regression_class, maxdegree= 10, method= "OLS", lmb= 0):
    from sklearn.utils import resample

    maxdegree = maxdegree + 1; start = 1; degrees = np.arange(start, maxdegree)
    error = np.zeros(maxdegree - start)
    bias = np.zeros(maxdegree - start)
    variance = np.zeros(maxdegree - start)
    n_bootstraps = 100
    
    if regression_class == "Scikit":
        for deg in degrees:
            print(f"----------------DEGREE: {deg}-----------------")
            scale_split = split.Scikit_split_scale(xx, yy, z_flat, deg= deg)
            
            boot_Z_tilde = np.zeros((scale_split.Z_train.shape[0], n_bootstraps))
            boot_Z_pred = np.zeros((scale_split.Z_test.shape[0], n_bootstraps))
            
            for i in range(n_bootstraps):
                X_, Z_ = resample(scale_split.X_train_scaled, scale_split.Z_train)
               
                Scikit_REG = reg.Scikit_regression(scale_split)
                Scikit_REG.lmb = lmb
                
                beta = Scikit_REG.fit(X_, Z_, method= method)
                Z_tilde = beta.predict(X_)
                Z_pred  = beta.predict(scale_split.X_test_scaled)
                   
                boot_Z_tilde[:,i]  = Z_tilde
                boot_Z_pred[:, i] = Z_pred
            
            error[deg - start] = np.mean( np.mean((scale_split.Z_test.reshape(-1,1) - boot_Z_pred)**2, axis= 1, keepdims= True) )
            bias[deg - start] = np.mean( (scale_split.Z_test.reshape(-1,1) - np.mean(boot_Z_pred, axis= 1, keepdims= True))**2 )
            variance[deg - start] = np.mean( np.var(boot_Z_pred, axis= 1, keepdims= True) )
        
        plt.figure()
        plt.title("Scikit Regression")
    else:
        for deg in degrees:
            print(f"----------------DEGREE: {deg}-----------------")
            scale_split = split.Numpy_split_scale(xx, yy, z_flat, deg= deg)
            
            boot_Z_tilde = np.zeros((scale_split.Z_train.shape[0], n_bootstraps))
            boot_Z_pred = np.zeros((scale_split.Z_test.shape[0], n_bootstraps))
            
            for i in range(n_bootstraps):
                X_, Z_ = resample(scale_split.X_train_scaled, scale_split.Z_train)
               
                Manual_REG = reg.Manual_regression(scale_split)
                Manual_REG.lmb = lmb
                
                beta = Manual_REG.fit(X_, Z_, method= method)
                Z_tilde   = X_ @ beta
                Z_pred    = scale_split.X_test_scaled @ beta
                
                boot_Z_tilde[:,i]  = Z_tilde
                boot_Z_pred[:, i] = Z_pred
                
            error[deg - start] = np.mean( np.mean((scale_split.Z_test.reshape(-1,1) - boot_Z_pred)**2, axis= 1, keepdims= True) )
            bias[deg - start] = np.mean( (scale_split.Z_test.reshape(-1,1) - np.mean(boot_Z_pred, axis= 1, keepdims= True))**2 )
            variance[deg - start] = np.mean( np.var(boot_Z_pred, axis= 1, keepdims= True) )
        
        plt.figure()
        plt.title("Manual Regression")
    
    plt.plot(degrees, error, label= "Error")
    plt.plot(degrees, bias, label= "Bias")
    plt.plot(degrees, variance, label= "Variance")
    plt.xlabel('Polynomial degree')
    plt.ylabel("Error/Bias/Variance")
    plt.legend()
    plt.show()
    
    
def cross_validation(scaler_obj, method = "OLS", k = 5, lmb= 0):
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score,\
        mean_squared_log_error, mean_absolute_error
    
    X = scaler_obj.X; Z = scaler_obj.Z
    X_scaled = scaler_obj.X_scaled
    
    # Initialize a KFold instance
    # kfold = KFold(n_splits = k, shuffle= True, random_state= 123)
    # kfold = KFold(n_splits = k, shuffle= True) # I'm not sure whether I should use shuffle or not
    kfold = KFold(n_splits = k)  
    print("k: ", k, " kfold: ", kfold)
    scores_KFold = np.zeros(k)
    
    Scikit_REG = reg.Scikit_regression(scaler_obj)
    Scikit_REG.lmb = lmb
    
    l = 0
    for train_idx, test_idx in kfold.split(X):
        # print("l= ", l)
        X_train = X_scaled[train_idx]; Z_train = Z[train_idx]
        X_test = X_scaled[test_idx]; Z_test = Z[test_idx]
        
        beta = Scikit_REG.fit(X_train, Z_train, method= method)
        
        Z_pred = beta.predict(X_test)
        
        scores_KFold[l] = np.sum((Z_pred - Z_test)**2)/np.size(Z_pred)
        
        l += 1
      
    Scikit_CV = cross_val_score(beta, X, Z, scoring='neg_mean_squared_error', cv=kfold)
    # Scikit_CV = cross_val_score(beta, X, Z, scoring='neg_mean_squared_error', cv=k)
    
    # print("Scikit_CV: ", -Scikit_CV, "\n", "scores_KFold: ", scores_KFold)
    # print(f"Scikit_CV: {-Scikit_CV} \n\nscores_KFold: {scores_KFold}")
    print(f"Scikit_CV: {np.mean(-Scikit_CV)} \n\nscores_KFold: {np.mean(scores_KFold)}")
    return np.mean(-Scikit_CV)

def deg_cross_validation(xx, yy, z_flat, regression_class, maxdegree= 10, method= "OLS", lmb= 0):
    maxdegree = maxdegree + 1; start = 1; degrees = np.arange(start, maxdegree)
    CV_MSE = np.zeros(maxdegree - start)
    time_CV = np.zeros(maxdegree - start)
    
    print("lmb: ", lmb)
    if regression_class == "Scikit":
        for deg in degrees:
            print(f"----------------DEGREE: {deg}-----------------")
            scale_split = split.Scikit_split_scale(xx, yy, z_flat, deg= deg)
            t0 = time.time()
            CV_MSE[deg - start] = cross_validation(scale_split, method = method, k = 10, lmb= lmb)
            time_CV[deg - start] = time.time() - t0
    
    else:
        for deg in degrees:
            print(f"----------------DEGREE: {deg}-----------------")
            scale_split = split.Numpy_split_scale(xx, yy, z_flat, deg= deg)
            t0 = time.time()
            CV_MSE[deg - start] = cross_validation(scale_split, method = method, k = 10, lmb= lmb)
            time_CV[deg - start] = time.time() - t0
    
    plt.figure()
    plt.title(method + f" Terrain Cross-validation $\lambda=$ {lmb}")
    # plt.title("Terrain Cross-validation OLS")
    plt.plot(degrees, np.log10(CV_MSE), "o-", label= "CV Test Error")
    plt.xlabel('Polynomial degree')
    plt.ylabel('log10[MSE]')
    plt.legend()
    plt.show()
    # save_fig(deg_cross_validation.__name__ + f"terrain{int(np.log10(lmb))}" + method)
    # save_fig(deg_cross_validation.__name__ + f"terrain" + method)
                
    
def Ridge_Lasso_MSE(xx, yy, z_flat, scaler_opt, regression_opt):
    if scaler_opt == "Numpy":
        scaler = split.Numpy_split_scale(xx, yy, z_flat, deg= 5)
    else:
        scaler = split.Scikit_split_scale(xx, yy, z_flat, deg= 5)
    
    nlambdas = 25
    lambdas = np.logspace(-4, 4, nlambdas)
    MSEpred = np.zeros(nlambdas); MSEtrain = np.zeros(nlambdas)
    MSELassotrain = np.zeros(nlambdas); MSELassopred = np.zeros(nlambdas)
    for i in range(nlambdas):
        lmb = lambdas[i]
        print(f"-------train_test_error-lmb-{lmb}-------")
        
        ErrorDict = train_test_error(scaler, regression_class= regression_opt, method = "Ridge", lmb = lmb, printToScreen= False)
        train_error, test_error = ErrorDict["MSE"]
        MSEtrain[i] = train_error; MSEpred[i] = test_error 
        
        ErrorDict = train_test_error(scaler, regression_class= "Scikit", method = "Lasso", lmb = lmb, printToScreen= False)
        train_error, test_error = ErrorDict["MSE"]
        MSELassotrain[i] = train_error; MSELassopred[i] = test_error
    
    plt.title(f"Ridge regression for deg = {scaler.deg}")
    plt.plot(np.log10(lambdas), MSEtrain, "o-", label='MSE Ridge Train')
    plt.plot(np.log10(lambdas), MSEpred, "o-", label='MSE Ridge Test')
    plt.plot(np.log10(lambdas), MSELassotrain, "o-", label='MSE Lasso Train')
    plt.plot(np.log10(lambdas), MSELassopred, "o-", label='MSE Lasso Test')
    plt.xlabel('log10($\lambda$)')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()
    # save_fig("RidgeLassoTrainTestMSE")

def R_L_model_complexity(xx, yy, z_flat, regression_opt = "Scikit", maxdegree= 10, method= "Ridge", lmb= 1, cv= False):
    train_vs_test_MSE(xx, yy, z_flat, regression_opt, maxdegree= maxdegree, method= method, lmb= lmb, cv= cv)
    plt.title(regression_opt + " " + method + f" train vs. test MSE for $\lambda=$ {lmb}")

def R_L_bias_variance(xx, yy, z_flat, regression_class = "Scikit", maxdegree= 10, method= "Ridge", lmb= 1):
    bias_variance_tradeoff(xx, yy, z_flat, regression_class = regression_class, maxdegree= maxdegree, method= method, lmb= lmb)
    plt.title(regression_class + " " + method + f" Bias-Variance tradeoff for $\lambda=$ {lmb}")

def lambda_loop(func_obj, xx, yy, z_flat, regression_class, maxdegree= 10,\
                method= "OLS", lmb= 0, start= -4, stop= 4, nlambdas= 9, var = 0, cv= False):
    
    print("lambda_loop Method: ", method)
    lambdas = np.logspace(start, stop, nlambdas)
    
    for i in range(nlambdas):
        lmb = lambdas[i]
        print("lmb", lmb)
        func_obj(xx, yy, z_flat, regression_class, maxdegree, method, lmb)
        plt.title(method + " " + func_obj.__name__ + f" $\lambda=$ {lmb}")
        save_fig(method + func_obj.__name__ + f"{int(np.log10(lmb))}--")



def plot_estimators(xx, yy, z_flat, regression_class= "Scikit", maxdegree= 10, method= "OLS", lmb= 0):
    maxdegree = maxdegree + 1; start = 1; degrees = np.arange(start, maxdegree)
    for deg in degrees:
        print(f"----------------DEGREE: {deg}-----------------")
        scaler= split.Scikit_split_scale(xx, yy, z_flat, deg= deg)
    
        Scikit_REG = reg.Scikit_regression(scaler)
        Scikit_REG.lmb = lmb
        beta = Scikit_REG.fit(scaler.X_train_scaled, scaler.Z_train, method= method)
        beta_array = np.copy(beta.coef_) 
        beta_array[0] += beta.intercept_
        beta = beta_array
        x = np.arange(len(beta))
        
        plt.figure()
        plt.title(method + f" estimators for a polynomial degree $p= {scaler.deg}$ and $\lambda= {lmb}$")
        plt.style.use("seaborn-darkgrid")
        plt.xticks(x)
        
        plt.plot(x, beta, ".", label=r'$\beta_j \pm 1.96 \sigma$')
        # plt.errorbar(x, beta, CI, fmt=".", capsize= 3, label=r'$\beta_j \pm 1.96 \sigma$')
        plt.legend(edgecolor= "black")
        plt.xlabel(r'index $j$')
        plt.ylabel(r'$\beta_j$')
        plt.show()
        # save_fig(method + plot_estimators.__name__ + f"{int(np.log10(lmb))}{deg}")

    
def regression(xx, yy, z_flat, method= "OLS"):
    if method == "Ridge":
        lmb= 0.001
        deg= 16 #[17, 19] or 16
    else: #Lasso
        lmb= 0.001
        deg= 14 #[12, 16] best 14
    
    if method == "OLS":
        scaler = split.Numpy_split_scale(xx, yy, z_flat, deg= 5)
        print("OLS:", scaler.deg)
        
        Scikit_REG = reg.Scikit_regression(scaler)
        Scikit_REG.lmb = lmb
        
        beta = Scikit_REG.fit(scaler.X_train_scaled, scaler.Z_train, method= method)
        Z = beta.predict(scaler.X_scaled)
    
    elif method == "Ridge":
        scaler = split.Numpy_split_scale(xx, yy, z_flat, deg= deg)
        print("Ridge:", scaler.deg)
        Scikit_REG = reg.Scikit_regression(scaler)
        Scikit_REG.lmb = lmb
        
        beta = Scikit_REG.fit(scaler.X_train_scaled, scaler.Z_train, method= method)
        Z = beta.predict(scaler.X_scaled)
    
    else: #Lasso
        scaler = split.Numpy_split_scale(xx, yy, z_flat, deg= deg)
        print("Lasso:", scaler.deg)
        Scikit_REG = reg.Scikit_regression(scaler)
        Scikit_REG.lmb = lmb
        
        beta = Scikit_REG.fit(scaler.X_train_scaled, scaler.Z_train, method= method)
        Z = beta.predict(scaler.X_scaled)
    
    return Z

def unflatten_z(z, z_flat):
    print(z.shape, z_flat.shape)
    
    Max = np.max(z_flat)
    Min = np.min(z_flat)
    for i in range(z_flat.shape[0]):
        if np.isnan(z_flat[i]):
            print(z_flat[i])
            z_flat[i] = 0
        elif np.isposinf(z_flat[i]):
            print(z_flat[i])
            z_flat[i] = Max
        elif np.isneginf(z_flat[i]):
            print(z_flat[i])
            z_flat[i] = Min
    
    print(Max, Min)
    array = np.zeros(z.shape)
    for i in range(z.shape[0]):
        # print("i: ", i)
        # array[i,:] = np.zeros(z.shape[1])
        array[i,:] = z_flat[ z.shape[1]*i : z.shape[1]*(i+1) ]
        # print(array[i] == z[i])
    return array
        
def terrain():
    import numpy as np
    from imageio import imread
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    import numpy as np
    from numpy.random import normal, uniform
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    # Load the terrain
    Dir = "../mohamahm/Data/"
    file= Dir + "SRTM_data_Norway_1.tif"
    terrain = imread(file)
    
    N = 200
    deg = 5 # polynomial order
    terrain = terrain[:N,:N]
    # Creates mesh of image pixels
    x = np.linspace(0,1, np.shape(terrain)[0])
    y = np.linspace(0,1, np.shape(terrain)[1])
    xx, yy = np.meshgrid(x,y)
    
    z = terrain
    z_flat = np.ravel(z)
    
    # X = create_X(x_mesh, y_mesh,m)
    # scale_split = split.Scikit_split_scale(xx, yy, z_flat, deg= deg)
    
    # fig, ax = plt.subplots()
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    
    # im = ax.imshow(terrain, cmap='bone')
    
    # fig.colorbar(im, cax=cax, orientation='vertical')
    # plt.show()

    # Show the terrain
    # fig = plt.figure()
    # plt.title('Terrain over Norway 1')
    # pos= plt.imshow(terrain, cmap='gray')
    # fig.colorbar(pos)
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.show()
    
    return xx, yy, z, z_flat


def save_fig(name):
    plt.savefig(name, dpi= 300, bbox_inches= 'tight')

def Exercise_4(xx, yy, z_flat, scaler_opt, regression_opt= "Scikit", maxdegree= 10, method= "Ridge", lmb= 1):
    
    #25 Degrees analysis lmb= 1000
    # R_L_model_complexity(xx, yy, z_flat, regression_opt = "Scikit", maxdegree= 25, method= method, lmb= 1000)
    # save_fig(Ridge_model_complexity.__name__ + f"{int(np.log10(1000))}" + "maxdegree25")
    # R_L_bias_variance(xx, yy, z_flat, regression_class = "Scikit", maxdegree= 25, method= method, lmb= 1000)
    # save_fig(Ridge_bias_variance.__name__ + f"{int(np.log10(lmb))}" + "maxdegree25" )
    
    
    nlambdas = 9
    lambdas = np.logspace(-4, 4, nlambdas)
    # cv_lmb = np.zeros((nlambdas, k))
    for i in range(nlambdas):
        lmb = lambdas[i]
        
        R_L_model_complexity(xx, yy, z_flat, regression_opt = "Scikit", maxdegree= 10, method= method, lmb= lmb, cv= True)
        # save_fig(method + R_L_model_complexity.__name__ + f"{int(np.log10(lmb))}")
        R_L_bias_variance(xx, yy, z_flat, regression_class = "Scikit", maxdegree= 10, method= method, lmb= lmb)
        # save_fig(Ridge_bias_variance.__name__ + f"{int(np.log10(lmb))}")

"""
# if func_obj.__name__ == confidence_interval.__name__:
#     scaler = split.Numpy_split_scale(xx, yy, z_flat, deg= 5)
#     # scaler = split.Scikit_split_scale(xx, yy, z_flat, deg= 5)
#     # func_obj(scaler, regression_class, var, method, lmb, errorbar= True)
#     for i in range(nlambdas):
#         lmb = lambdas[i]
#         print("lmb", lmb)
#         func_obj(scaler, regression_class, var, method, lmb, errorbar= True)
#         plt.title(method + " " + func_obj.__name__ + f" $\lambda=$ {lmb}")
#         save_fig(method + func_obj.__name__ + f"{int(np.log10(lmb))}")
    
#     sys.exit(0)
"""