import GD as gd
import regression as reg
import error as err
from functions import *

# =============================================================================
# Python Libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

dataset = load_breast_cancer()
data = dataset.data
targets = dataset.target

test_size = 0.2
train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size= test_size)
test_targets = test_targets[:, np.newaxis]

N= data.shape[0]
p= data.shape[1]
N_train = train_data.shape[0]
N_test = test_data.shape[0]


Log_REG = reg.Logistic_regression()
# GD= gd.Gradient_descent(train_data, train_targets, Log_REG)

stops1, ACC_cache1, learning_rates1, batch_sizes1 =\
    analysis(train_data, train_targets, test_data, test_targets, Log_REG, method= "Logistic Regression", analysis= "batch_size", plot_opt= True)

plt.close("all")
make_heat_map(stops1/stops1.max(), batch_sizes1, np.log10(learning_rates1), vmin= 0, vmax= 1)
title_string = "Logistic Regression \n Epochs elapsed before SGD converges; 1 equating to 200000 epochs."
plt.title(title_string, fontsize= 12)
plt.xlabel("Batch size"); plt.ylabel(r"$\log(\eta)$")
save_fig("../Figures/LogReg-Epochs.png")
plt.show()

make_heat_map(ACC_cache1, batch_sizes1, np.log10(learning_rates1), fmt = ".3f")
title_string = "Logistic Regression \n Accuracy as a function of the learning rate and batch size"
plt.title(title_string, fontsize= 12)
plt.xlabel("Batch size"); plt.ylabel(r"$\log(\eta)$")
save_fig("../Figures/LogREg-ACC.png")
plt.show()


# Determining the best batch size for L2
col_mean = np.mean(ACC_cache1, axis= 0)
optimal_idx = np.where(col_mean == col_mean.max())[0][0]
optimal_batch_size = batch_sizes1[optimal_idx]


# # L2-regularization
stops2, MSE_cache2, learning_rates2, lmbs =\
    analysis(train_data, train_targets, test_data, test_targets, Log_REG, method= "L2-regularization", analysis= "hyperparameter", plot_opt= True, batch_size= optimal_batch_size)


make_heat_map(stops2/stops2.max(), np.log10(lmbs), np.log10(learning_rates2), vmin= 0, vmax= 1)
title_string = "L2 Logistic Regression \n Epochs elapsed before SGD converges; 1 equating to 200000 epochs."
plt.title(title_string, fontsize= 12)
plt.xlabel(r"$\log(\lambda)$"); plt.ylabel(r"$\log(\eta)$")
save_fig("../Figures/L2-Epochs.png")
plt.show()

make_heat_map(MSE_cache2, np.log10(lmbs), np.log10(learning_rates1), fmt = ".3f")
title_string = "L2 Logistic Regression \n Accuracy as a function of the learning rate and hyperparameter $\lambda$"
plt.title(title_string, fontsize= 12)
plt.xlabel(r"$\log(\lambda)$"); plt.ylabel(r"$\log(\eta)$")
save_fig("../Figures/L2-ACC.png")
plt.show()


#### Depricated ####

# #test
# pred= Log_REG.predict(test_data, SGD_weights)
# pred= np.around(pred).astype(np.int64)
# for i in range(len(test_targets)):
#     print(pred[i][0], test_targets[i])

# def Stochastic_GD( initial_guess, method= "OSL", batch_size= 5,\
                  # epochs= 30, learning_rate= None, plotting= False):
# weights = GD.start_beta
# GD_weights = GD.Simple_GD(GD.start_beta, method= "Logistic Regression", Niterations= N)

# #test
# pred= logistic_prediction(GD_weights, test_data)

# for i in range(len(test_targets)):
    # print(pred[i][0], test_targets[i])


# print(pred[])

# print(sigmoid(data))

# def sigmoid(x):
    # return 0.5 * (np.tanh(x / 2.) + 1)

# def sigmoid(x):
#     return 1/(1 + np.exp(x))

# def logistic_predictions(weights, inputs):
#     # Outputs probability of a label being true according to logistic model.
#     return sigmoid(np.dot(inputs, weights))

# def training_loss(weights):
#     # Training loss is the negative log-likelihood of the training labels.
#     preds = logistic_predictions(weights, inputs)
#     label_probabilities = preds * targets + (1 - preds) * (1 - targets)
#     return -np.sum(np.log(label_probabilities))

# # Build a toy dataset.
# inputs = data
# targets = targets

# # Define a function that returns gradients of training loss using Autograd.
# training_gradient_fun = grad(training_loss)

# # Optimize weights using gradient descent.
# weights = np.zeros(p)
# print("Initial loss:", training_loss(weights))
# for i in range(100):
#     weights -= training_gradient_fun(weights) * 0.01

# print("Trained loss:", training_loss(weights))