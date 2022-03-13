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

# =============================================================================
# To-do's
# Exercise 1 Confidence interval use beta = sigma*(X_T*X)^-1, talk about why we scaled the data
# Exercise 2 Fig 2.11 Hastie, Tibshirani (test and training MSE), bias-variance tradeoff(using bootstrap)
# Exercise 3 Cross-validation
# =============================================================================

# Presets
mean = 0; deviation = 0.1;  #Franke
random = True; precision = 0.05 #initialize_array
noise = True #franke_function

# Generating data using the Franke function
Fr= fr.Franke(mean, deviation)

x, y = Fr.initialize_array(random, precision)
# x, y = Fr.initialize_array(random = True, precision= 0.01)
xx, yy = np.meshgrid(x, y)

plt.close("all")
plt.style.use("seaborn-darkgrid")

z = Fr.franke_function(xx, yy, noise = False)
# Fr.plot_contour(xx, yy, z)
# Fr.plot_3D(xx, yy, z)

z_noisey, var = Fr.franke_function(xx, yy, noise)
# Fr.plot_contour(xx, yy, z_noisey)
# Fr.plot_3D(xx, yy, z_noisey)

z_flat = np.ravel(z_noisey)
# z_flat = z_noisey.reshape((-1,1))
# -------------------------------------------------------------------------------------

from Exercises import *

print("Choose exercise! Options [\"Exercise1\", \"Exercise2\", \"Exercise3\", \"Exercise4\", \"Exercise5\"]")
# exercise = str(input()) # Provide an input
exercise = "Exercise6" # or change this line and comment out the one above

scaler_opt = "Numpy" 
# scaler_opt = "Scikit"

# regression_opt = "Manual" 
regression_opt = "Scikit"

method = "OLS"
# method = "Ridge"
# method = "Lasso"

if exercise in ["Exercise1", "Exercise 1"]:
    #Exercise1
    """
    
    Choose the scaler class Numpy/Scikit. Choose the regression class Manual/Scikit
    Note: Both train_test_error and confidence_interval takes an optional parameter
    "method", defaul = "OLS". 
    Manual does OLS and Ridge. Scikit does OLS, Ridge and Lasso.
    
    """
    if scaler_opt == "Numpy":
        scaler = split.Numpy_split_scale(xx, yy, z_flat, deg= 5)
    else:
        scaler = split.Scikit_split_scale(xx, yy, z_flat, deg= 5)
    
    print(xx.shape, yy.shape, z_flat.shape, z_noisey.shape)
    
    print("-------train_test_error--------")
    train_test_error(scaler, regression_class= regression_opt, method = "OLS", printToScreen= True, defaultError= "Scikit")
    # train_test_error(scaler, regression_class= regression_opt, method = "OLS", printToScreen= True, defaultError= "Manual")
    print("-------Confidence_interval---------")
    confidence_interval(scaler, regression_class= regression_opt, var= var, method = "OLS")
    # save_fig(confidence_interval.__name__ + regression_opt + scaler_opt)
    
elif exercise in ["Exercise2", "Exercise 2"]:
    # Exercise2 
    """
    
    Choose the regression class Manual/Scikit. If "Manual" scaling is done using
    Numpy_split_scale then manual regression, options OLS and Ridge available.
    Otherwise if "Scikit" is selected then everything is done using Scikit.
    
    """
    
    print("Method: ", method, exercise)
    print("---------train_vs_test_MSE----------")
    train_vs_test_MSE(xx, yy, z_flat, regression_class= regression_opt, maxdegree= 20)
    # save_fig(train_vs_test_MSE.__name__ + regression_opt)
    print("---------bias_variance_tradeoff---------") 
    bias_variance_tradeoff(xx, yy, z_flat, regression_class= regression_opt, maxdegree= 8) #0.05, 0.02, 0.01 : maxdegree = 8, 15/18, 20
    # plt.title(regression_opt + f" Regression N= {Fr.steps}" )
    # save_fig(bias_variance_tradeoff.__name__ + regression_opt + f"N= {Fr.steps}")

elif exercise in ["Exercise3", "Exercise 3"]:
    """
    Scaling con be done usinig scikit or numpy. Prints out CV
    using both a KFold-algorithm and cross_val_score from Scikit.
    """
    
    maxdegree = 6
    train_vs_test_MSE(xx, yy, z_flat, regression_class= regression_opt, maxdegree= maxdegree, method= "OLS", lmb= 0, cv= True)
    # save_fig(train_vs_test_MSE.__name__ + regression_opt + f"maxdegree={maxdegree}")

elif exercise in ["Exercise4", "Exercise 4"]:
    
    Exercise_4(xx, yy, z_flat, scaler_opt, regression_opt= "Scikit", maxdegree= 10, method= "Ridge")
    
elif exercise in ["Exercise5", "Exercise 5"]:
    method= "Lasso"
    maxdegree = 10
    
    # lambda_loop(plot_estimators, xx, yy, z_flat, regression_class= regression_opt, maxdegree= 10,\
                # method= method)
    
    lambda_loop(bias_variance_tradeoff, xx, yy, z_flat, regression_class= regression_opt, maxdegree= maxdegree,\
                method= method)
        
    # lambda_loop(deg_cross_validation, xx, yy, z_flat, regression_class= regression_opt, maxdegree= maxdegree,\
                # method= method, lmb= None, start= -4, stop= 4, nlambdas= 9)

elif exercise in ["Exercise6", "Exercise 6"]:
    xx, yy, z, z_flat = terrain()
    # method= "OLS"
    method= "Ridge"
    # method= "Lasso"
    maxdegree= 10
    print(xx.shape, yy.shape, z.shape, z_flat.shape)
    # Exercise 1
    # scaler = split.Numpy_split_scale(xx, yy, z_flat, deg= 5)
    # var = np.var(scaler.X[:,1:])
    # if scaler_opt == "Numpy":
    #     scaler = split.Numpy_split_scale(xx, yy, z_flat, deg= 5)
    # else:
    #     scaler = split.Scikit_split_scale(xx, yy, z_flat, deg= 5)
    
    # confidence_interval(scaler, regression_class= regression_opt, var= var, method = "OLS", errorbar= True)
    
    # Exercise 2
    
    print("---------train_vs_test_MSE----------")
    # train_vs_test_MSE(xx, yy, z_flat, regression_class= regression_opt, maxdegree= 10)
    # save_fig(train_vs_test_MSE.__name__ + regression_opt + "terrain")
    print("---------bias_variance_tradeoff---------") 
    # bias_variance_tradeoff(xx, yy, z_flat, regression_class= regression_opt, maxdegree= 8) #0.05, 0.02, 0.01 : maxdegree = 8, 15/18, 20
    # plt.title(regression_opt + f" Regression N= {Fr.steps}" )
    # save_fig(bias_variance_tradeoff.__name__ + regression_opt + f"N={Fr.steps}" + "terrain")
    
    #OLS
    # maxdegree = 10
    # train_vs_test_MSE(xx, yy, z_flat, regression_class= regression_opt, maxdegree= maxdegree, method= "OLS", lmb= 0, cv= True)
    # deg_cross_validation(xx, yy, z_flat, regression_class= regression_opt, maxdegree= maxdegree, method= "OLS")
    
    
    # Ridge/Lasso
    # lambda_loop(deg_cross_validation, xx, yy, z_flat, regression_class= regression_opt, maxdegree= maxdegree,\
                # method= method, lmb= None, start= -4, stop= 1, nlambdas= 6)
    
    # Z = regression(xx, yy, z_flat, method= method)
    # Z_pred= unflatten_z(z, Z) #z(200,200) Z(40000) --> (200, 200)
    # print(xx.shape, yy.shape, Z.shape, Z_pred.shape)
    # Fr.plot_3D(xx, yy, Z_pred, title= "erer 331") 
    
    """
    We wanted to plot but we couldn't find out why
    matplotlib never rendered our plots. We thought it's
    either NaN, inf or -inf. But it doesn't seem like it.
    """
    
if __name__ == "__main__":
    print("Running main.py")







"""
from Exercises import Exercise1, Exercise2

# Exercise1(x, y, z_noisey, var)

# degree_analysis = Exercise2(x, y, z_noisey)
degree_analysis = Exercise2(xx, yy, z_flat)


#Bootstrap
# scale_split = split.Numpy_split_scale(x, y, z_noisey)
scale_split = split.Scikit_split_scale(xx, yy, z_flat)
n_bootstraps = 100
Res = res.Resampling(scale_split)
Res.Bootstrap(regression_class= "Scikit", fit_method = "OLS", n_bootstraps = n_bootstraps, printToScreen= True)
Res.Bootstrap(regression_class= "Manual", fit_method = "OLS", n_bootstraps = n_bootstraps, printToScreen= True)
Res.Scikit_CrossValidation(fit_method = "OLS")
"""
    
"""
# del F; del FF

# print(F.__class__.__name__)
# print("wowo ", F.__class__)

# print("=====================")
# print(FAM.__class__.__name__)
# print("wowo ", FAM.__class__)

"""