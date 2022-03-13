import GD as gd
import error as err

# =============================================================================
# Python Libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analysis(X_train, Z_train, X_test, Z_test, reg_obj, method= "OLS", analysis= "batch_size", plot_opt = False, batch_size= None):
    GD = gd.Gradient_descent(X_train, Z_train, reg_obj)
    # learning_rates = np.logspace(-4, -1, 4) # 0.1 could result in overflow
    learning_rates = np.logspace(-4, -2, 3)
    
    if batch_size == None:
        if method in ["ols", "OLS", "Ols", "ridge", "Ridge", "RIDGE"]:
            batch_sizes = [20, 40, 80, 160] # Franke
        else:
            batch_sizes = [13, 35, 65, 91]  # Logistic regression cancer [5, 7, 13, 35, 65, 91]
    else:
        batch_size = batch_size
    
    Err_pred = err.Error(Z_test)
    if method in ["ols", "OLS", "Ols", "ridge", "Ridge", "RIDGE"]:
        error_func = Err_pred.MSE
    else:
        error_func = Err_pred.Accuracy
        
    
    # epochs_num = int(2*1e2)
    epochs_num = int(2*1e5)
    
    print(X_test.shape, Z_test.shape)
    
    if analysis in ["batch_size", "batch size", "Batch size"]:
        print(f"Batch size analysis initiated, sizes: {batch_sizes}")
        error_cache = np.zeros((len(learning_rates), len(batch_sizes)))
        stops = error_cache.copy()
        for i, learning_rate in enumerate(learning_rates):
            for j, batch_size in enumerate(batch_sizes):
                SGD_beta, error_train, stop = GD.Stochastic_GD(GD.start_beta, method= method, batch_size= batch_size,\
                                                        epochs= epochs_num, learning_rate= learning_rate, plotting= plot_opt)
                # print("SGD OLS: ", SGD_beta.ravel())
                stops[i][j] = stop # At which epoch did we stop
                Z_pred = reg_obj.predict(X_test, SGD_beta)
                error_pred = error_func(Z_test, Z_pred)
                error_cache[i][j] = error_pred
                # print(f"eta: {learning_rate}, batch_size: {batch_size}, {MSE_pred}, {MSE_cache[i][j]}")
        
        return stops, error_cache, learning_rates, batch_sizes
    
    elif analysis in ["hyperparameter", "hyper", "HYPER"]:
        lmbs = np.logspace(-5, -1, 5) # lambda
        print(f"Hyperparameter analysis initiated, values: {lmbs}")
        error_cache = np.zeros((len(learning_rates), len(lmbs)))
        stops = error_cache.copy()
        for i, learning_rate in enumerate(learning_rates):
            for j, lmb in enumerate(lmbs):
                GD.lmb = lmb # lambda
                SGD_beta, MSE_train, stop = GD.Stochastic_GD(GD.start_beta, method= method, batch_size= batch_size,\
                                                        epochs= epochs_num, learning_rate= learning_rate, plotting= plot_opt)
                # print("SGD OLS: ", SGD_beta.ravel())
                stops[i][j] = stop # At which epoch did we stop
                Z_pred = reg_obj.predict(X_test, SGD_beta)
                error_pred = error_func(Z_test, Z_pred)
                error_cache[i][j] = error_pred
        
        return stops, error_cache, learning_rates, lmbs
    
def make_heat_map(matrix, x_axis, y_axis, fmt= ".2f", vmin= None, vmax= None):
    plt.figure()
    
    if fmt[-1] == "f":
        np.around(matrix, int(fmt[-2]))
    
    
    if vmin == None and vmax == None:
        vmin = matrix.max()
        vmax = matrix.min()
    
    ax = sns.heatmap(matrix, xticklabels= x_axis, yticklabels= y_axis,\
                     annot= True, fmt= fmt, vmin= vmin, vmax= vmax)
    # if vmin != None and vmax != None:
    #     ax = sns.heatmap(matrix, xticklabels= x_axis, yticklabels= y_axis,\
    #                      annot= True, fmt= fmt, vmin= vmin, vmax= vmax)
    # else:
    #     ax = sns.heatmap(matrix, xticklabels= x_axis, yticklabels= y_axis,\
    #                      annot= True, fmt= fmt)    
    
def save_fig(name):
    plt.savefig(name, dpi= 180, bbox_inches= 'tight')