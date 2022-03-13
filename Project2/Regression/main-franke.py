import frankefunction as fr
import split_scale as scaler
import regression as reg
import GD as gd
import error as err
from functions import *

# =============================================================================
# Python Libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.linear_model import SGDRegressor #testing
# import seaborn as sns


# Presets
mean = 0; deviation = 0.1;  #Franke
random = True; precision = 0.05 #initialize_array
noise = True #franke_function

# Generating data using the Franke function
Fr= fr.Franke(mean, deviation)

x, y = Fr.initialize_array(random, precision)
xx, yy = np.meshgrid(x, y)

plt.close("all")
plt.style.use("seaborn-darkgrid")

z = Fr.franke_function(xx, yy, noise = False)
# Fr.plot_contour(xx, yy, z)
# Fr.plot_3D(xx, yy, z)

z_noisey, var = Fr.franke_function(xx, yy, noise)
# Fr.plot_contour(xx, yy, z_noisey)
# Fr.plot_3D(xx, yy, z_noisey)

# z_flat = np.ravel(z_noisey)
z_flat = z_noisey.ravel()
# z_flat = z_noisey.reshape((-1,1))
# -------------------------------------------------------------------------------------

# Generating test case
scaler = scaler.Numpy_split_scale(xx, yy, z_flat, deg= 5)
print("scaler: ", scaler.X.shape, scaler.Z.shape)

X_train = scaler.X_train_scaled; Z_train = scaler.Z_train
X_test = scaler.X_test_scaled; Z_test = scaler.Z_test; Z_test = Z_test[:,np.newaxis]

Manual_REG = reg.Manual_regression(scaler)
Manual_REG.lmb = 0
beta_OLS = Manual_REG.fit(X_train, Z_train, method= "OLS")
print("OLS: ", beta_OLS, "\n")

GD = gd.Gradient_descent(X_train, Z_train, Manual_REG)
# SGD_beta_OLS, train_MSE, stop = GD.Stochastic_GD(GD.start_beta, method= "OSL", batch_size= 20,\
                  # epochs= 10000, learning_rate= 0.1, plotting= True)


"""
# Testing for Gamma
gammas = np.linspace(0.1, 0.9, 9)
epochs_num = int(2*1e5)
plt.close("all")
plt.figure()
plt.title(f"Training MSE of SGD OLS as a fucntion of the momentum parameter $\gamma$")
stops = []
for i in gammas:
    gamma = np.around(i,1)
    GD.gamma = gamma
    SGD_beta_OLS, MSE, stop = GD.Stochastic_GD(GD.start_beta, method= "OLS", batch_size= 40,\
                                            epochs= epochs_num, learning_rate= 0.01, plotting= True)
    MSE = MSE[1:] # Exculding the first value of the MSE since it's calculated using the random initial guess
    stops.append(stop)
    plt.plot(np.arange(len(MSE)), np.log10(MSE), label= f"Stop Epoch: {stop}, $\gamma=$ {gamma}")


# plt.xlim(30, max(stops))
plt.xlim(-10, min(stops))
plt.xlabel("Epoch"); plt.ylabel("log10[MSE]")
plt.legend()
save_fig("../Figures/gamma_analysis.png")
plt.show()
"""

"""
# SGD Momentum
# OLS
stops1, MSE_cache1, learning_rates1, batch_sizes1 =\
    analysis(X_train, Z_train, X_test, Z_test, Manual_REG, method= "OLS", analysis= "batch_size", plot_opt= True)

plt.close("all")
make_heat_map(stops1/stops1.max(), batch_sizes1, np.log10(learning_rates1), vmin= 0, vmax= 1)
title_string = "OLS \n Epochs elapsed before SGD converges; 1 equating to 200000 epochs."
plt.title(title_string, fontsize= 12)
plt.xlabel("Batch size"); plt.ylabel(r"$\log(\eta)$")
# save_fig("../Figures/OLS-Epochs.png")
plt.show()

make_heat_map(MSE_cache1, batch_sizes1, np.log10(learning_rates1), fmt = ".3f")
title_string = "OLS \n MSE as a function of the learning rate and batch size"
plt.title(title_string, fontsize= 12)
plt.xlabel("Batch size"); plt.ylabel(r"$\log(\eta)$")
# save_fig("../Figures/OLS-MSE.png")
plt.show()

# Determining the best batch size for Ridge
col_mean = np.mean(MSE_cache1, axis= 0)
optimal_idx = np.where(col_mean == col_mean.min())[0][0]
optimal_batch_size = batch_sizes1[optimal_idx]

# Ridge
stops2, MSE_cache2, learning_rates2, lmbs =\
    analysis(X_train, Z_train, X_test, Z_test, Manual_REG, method= "Ridge", analysis= "hyperparameter", plot_opt= True, batch_size= optimal_batch_size)


make_heat_map(stops2/stops2.max(), np.log10(lmbs), np.log10(learning_rates2), vmin= 0, vmax= 1)
title_string = "Ridge \n Epochs elapsed before SGD converges; 1 equating to 200000 epochs."
plt.title(title_string, fontsize= 12)
plt.xlabel(r"$\log(\lambda)$"); plt.ylabel(r"$\log(\eta)$")
save_fig("../Figures/Ridge-Epochs.png")
plt.show()

make_heat_map(MSE_cache2, np.log10(lmbs), np.log10(learning_rates1), fmt = ".3f")
title_string = "Ridge \n MSE as a function of the learning rate and hyperparameter $\lambda$"
plt.title(title_string, fontsize= 12)
plt.xlabel(r"$\log(\lambda)$"); plt.ylabel(r"$\log(\eta)$")
save_fig("../Figures/Ridge-MSE.png")
plt.show()
"""

#### Depricated ####

# optimal_parameter = np.where(MSE_cache == MSE_cache.min())
# optimal_learning_rate = optimal_parameter[0][0]
# optimal_batch_size = optimal_parameter[1][0]

# Plotting
# GD_beta_OLS, MSE_1, stop1 = GD.Simple_GD(GD.start_beta, method= "OLS", Niterations= 200000, plotting= True)
# print("GD OLS: ", GD_beta_OLS.ravel())

# GD.lmb = 0.0001
# GD.lmb = 0.001
# GD_beta_Ridge, MSE_2, stop2 = GD.Simple_GD(GD.start_beta, method= "Ridge", Niterations= 200000, plotting= True)
# print("GD Ridge: ", GD_beta_Ridge.ravel())

# print("Before: ", MSE_1[0], MSE_2[0])
# MSE_1 = MSE_1[1:]; MSE_2 = MSE_2[1:]
# print("After: ", MSE_1[0], MSE_2[0],)

# plt.close("all")

# plt.figure()
# plt.title("Gradient Descent MSE")
# plt.plot(np.arange(len(MSE_1)), np.log10(MSE_1), label= "GD OLS")
# plt.plot(np.arange(len(MSE_2)), np.log10(MSE_2), label= "GD Ridge")
# plt.xlabel("Iteration"); plt.ylabel("log10[MSE]")
# plt.legend()
# plt.show()

# DOESN'T WORK
# epochs_num = int(2*1e1)
# epochs_num = int(2*1e5)
# GD.gamma = 0.3
# SGD_beta_OLS, MSE_3, stop3 = GD.Stochastic_GD(GD.start_beta, method= "OLS", batch_size= 40,\
#                                         epochs= epochs_num, learning_rate= 0.01, plotting= True)
# print("SGD OLS: ", SGD_beta_OLS.ravel())

# GD.lmb = 0.0001
# GD.lmb = 0.001
# SGD_beta_Ridge, MSE_4, stop4 = GD.Stochastic_GD(GD.start_beta, method= "Ridge", batch_size= 40, epochs= epochs_num, plotting= True)
# print("SGD Ridge: ", SGD_beta_Ridge.ravel())

# Exculding the first value of the MSE since it's calculated using the random initial guess
# print("Before: ", MSE_1[0], MSE_2[0], MSE_3[0], MSE_4[0])
# MSE_1 = MSE_1[1:]; MSE_2 = MSE_2[1:]; MSE_3 = MSE_3[1:]; MSE_4 = MSE_4[1:];
# print("After: ", MSE_1[0], MSE_2[0], MSE_3[0], MSE_4[0])




# plt.figure()
# plt.title("Stochastic Gradient Descent MSE")
# plt.plot(np.arange(len(MSE_3)), np.log10(MSE_3), label= "SGD OLS")
# plt.plot(np.arange(len(MSE_4)), np.log10(MSE_4), label= "Ridge OLS")
# plt.xlabel("Epoch"); plt.ylabel("log10[MSE]")
# plt.legend()
# plt.show()
