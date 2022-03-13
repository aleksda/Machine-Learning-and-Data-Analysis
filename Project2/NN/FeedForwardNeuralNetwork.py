import numpy as np
import autograd.numpy as np
from autograd import elementwise_grad
import matplotlib.pyplot as plt
import seaborn as sns

class NeuralNetwork:
    #### Neural network using SGD with momentum
    def __init__(self,
                 X,
                 y,
                 nr_of_hidden_layers  = 1,
                 nr_of_hidden_nodes   = 10,
                 batch_size           = 4,
                 eta                  = 0.001,
                 lmbd                 = 0.00,
                 gamma                = 0.0,        # For eta-decay
                 cost                 = "MSE",
                 activation           = "sigmoid",
                 last_activation      = None):

        self.X              = X
        self.in_n           = X.shape[0]
        self.num_features   = X.shape[1]

        self.y              = y                    # Target
        self.Y              = np.copy(y)
        self.num_categories = y.shape[1]

        self.nr_of_hidden_layers = nr_of_hidden_layers
        self.nr_of_hidden_nodes  = nr_of_hidden_nodes
        self.nr_of_output_nodes  = self.num_categories

        self.activation = activation
        self.batch_size = batch_size
        self.lmbd       = lmbd
        self.gamma      = gamma

        self.e          = eta
        self.A          = 2
        self.k          = 0
        self.dt         = 0

        self.tol           = 1e-8
        self.scaled_weight = [1, 1, 1]
        self.tot_layers    = self.nr_of_hidden_layers+2

        self.data_indices  = np.arange(self.in_n)

        sig  = ["sigmoid", "Sigmoid"]
        relu = ["ReLU", "Relu", "relu"]
        lelu = ["Leaky_ReLU", "Leaky_relu", "Leaky_Relu", "leaky_relu"]
        soft = ["Softmax", "softmax"]

        if self.activation in sig:
            activate           = self.sigmoid
            activate_der       = self.sigmoid_der

        elif self.activation in relu:
            activate           = self.ReLU
            activate_der       = self.ReLU_der
            self.scaled_weight = [nr_of_hidden_nodes**2, nr_of_hidden_nodes * self.num_features, 
                                  self.nr_of_hidden_nodes * nr_of_hidden_nodes] # maybe change index 1 and 2

        elif self.activation in lelu:
            activate           = self.Leaky_ReLU
            activate_der       = self.Leaky_ReLU_der
            self.scaled_weight = [nr_of_hidden_nodes**2, nr_of_hidden_nodes * self.num_features, 
                                  self.nr_of_hidden_nodes * nr_of_hidden_nodes] # maybe change index 1 and 2

        elif self.activation in soft:
            activate           = self.soft_max
            activate_der       = self.soft_max_der

        else:
            raise ValueError(f"Input must be either 'Sigmoid', 'ReLU', 'Leaky_ReLU', or 'Softmax', not '{self.activation}'.")

        if last_activation in sig:
            last_act          = self.sigmoid
            self.last_act_der = self.sigmoid_der

        elif last_activation in relu:
            last_act          = self.ReLU
            self.last_act_der = self.ReLU_der

        elif last_activation in lelu:
            last_act          = self.Leaky_ReLU
            self.last_act_der = self.Leaky_ReLU_der

        elif last_activation in soft:
            last_act          = self.soft_max
            self.last_act_der = self.soft_max_der

        elif last_activation == None:
            last_act          = activate
            self.last_act_der = activate_der

        self.act_list      = []
        self.act_list_der  = []
        for a in range(self.tot_layers):
            self.act_list.append(activate)
            self.act_list_der.append(activate_der)

        if last_activation is not None:
            self.act_list[-1] = last_act

        reg = ["MSE", "Mse", "mse"]
        cla = ["Cross_Entropy", "Cross_entropy", "cross_entropy", "Cross Entropy", "Cross entropy", "cross entropy"]

        if cost in reg:
            self.cost     = self.MSE
            self.cost_der = self.MSE_der

        elif cost in cla:
            self.cost     = self.cross_entropy
            self.cost_der = self.cross_entropy_der

        else:
            raise ValueError(f"Input must be either 'MSE', or 'Cross_Entropy', not'{cost}'.")

        if self.cost == self.MSE:
            self.score_shape = 1
            self.allowed_acc = False

        elif self.cost == self.cross_entropy:
            self.score_shape = self.num_categories
            self.allowed_acc = True

        self.create_layers()
        self.create_weights_biases()

    def create_layers(self):

        self.a = []
        self.a.append(self.X.copy())                                                             # Input
        for i in range(self.nr_of_hidden_layers):
            self.a.append(np.zeros([self.in_n, self.nr_of_hidden_nodes], dtype=np.float64)) # Hidden
        self.a.append(np.zeros((np.shape(self.y)), dtype=np.float64))                       # Output

        self.z = self.a.copy()

    def create_weights_biases(self):

        bias_shift = 0.1

        nr_of_hidden_layers = self.nr_of_hidden_layers
        num_features        = self.num_features
        nr_of_hidden_nodes  = self.nr_of_hidden_nodes
        num_categories      = self.num_categories

        self.weights = []
        self.weights.append(np.nan)
        self.weights.append(np.random.randn(nr_of_hidden_nodes, num_features) / (self.scaled_weight[0]))
        for w in range(nr_of_hidden_layers - 1):
            self.weights.append(np.random.randn(nr_of_hidden_nodes, nr_of_hidden_nodes) / (self.scaled_weight[1]))
        self.weights.append(np.random.randn(self.nr_of_output_nodes, nr_of_hidden_nodes) / (self.scaled_weight[2]))

        self.bias = []
        self.bias.append(np.nan)
        for i in range(nr_of_hidden_layers):
            self.bias.append(np.ones(nr_of_hidden_nodes) * bias_shift)
        self.bias.append(np.ones(num_categories) * bias_shift)

        self.weights_v = []
        self.bias_v = []
        self.weights_v.append(np.nan)
        self.bias_v.append(np.nan)
        for i in range(1, nr_of_hidden_layers + 2):
            self.weights_v.append(np.zeros(np.shape(self.weights[i])))
            self.bias_v.append(np.zeros(np.shape(self.bias[i])))

        # Error term
        self.local_gradient    = self.a.copy()
        self.local_gradient[0] = np.nan

    def feed_forward(self):
        for wb in range(1, self.tot_layers):

            Z_wb = self.a[wb-1] @ self.weights[wb].T + self.bias[wb][np.newaxis, :]
            self.z[wb] = Z_wb
            self.a[wb] = self.act_list[wb](Z_wb)

    def backpropagation(self):
        self.local_gradient[-1] = self.cost_der(self.a[-1]) * self.last_act_der(self.z[-1])
        self.check_grad = np.linalg.norm(self.local_gradient[-1]   * self.eta)

        for i in range(self.tot_layers - 2, 0, -1):
            self.local_gradient[i] = self.local_gradient[i+1] @ self.weights[i+1] * self.act_list_der[i](self.z[i])

    def update_parameters(self):
        self.backpropagation()

        if self.check_grad > self.tol and np.isfinite(self.check_grad):

            for i in range(1, self.tot_layers):
                self.weights_v[i] = self.gamma * self.weights_v[i] +\
                self.eta * (self.local_gradient[i].T @ self.a[i-1] + self.weights[i] * self.lmbd)
                self.weights[i] -= self.weights_v[i]

                self.bias_v[i] = self.gamma * self.bias_v[i] + self.eta * np.mean(self.local_gradient[i], axis=0) +\
                self.lmbd * self.bias[i]
                self.bias[i] -= self.bias_v[i]

    def predict(self, X):

        self.a[0] = X
        self.feed_forward()
        return self.act_list[-1](self.z[-1])

    def train(self, epochs):

        self.score = np.zeros([epochs+1, self.score_shape])
        self.check_grad = 1
        epoch = 0

        while epoch < epochs and self.check_grad > self.tol and np.isfinite(self.check_grad):

            self.eta_func(epoch)
            batches = self.__get_batches()
            epoch += 1
            for batch in batches:
                self.a[0] = self.X[batch]
                self.y    = self.Y[batch]
                self.feed_forward()
                self.update_parameters()

        self.num_epochs = epoch

    def eta_func(self, epoch):
        self.eta = self.e * self.A * self.sigmoid(self.k * (self.dt - epoch))

    def __get_batches(self):
        idx = np.arange(self.in_n)
        np.random.shuffle(idx)

        int_max = self.in_n // self.batch_size
        batches = [idx[i*self.batch_size:(i+1) * self.batch_size] for i in range(int_max)]

        if self.in_n % self.batch_size != 0:
            batches.append(idx[int_max*self.batch_size:])

        return batches

    ################################ SETTERS AND GETTERS ################################
    def set_eta_decay(self, k, dt):
        self.k  = k
        self.dt = dt
        self.A  = 1 / self.sigmoid(self.k * self.dt)

    def set_eta(self, eta):
        self.eta = eta

    def get_nr_of_epochs(self):
        return self.num_epochs

    ################################ SETTERS AND GETTERS ################################

    #################################### ACTIVATION ####################################
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_der(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def ReLU(self, z):
        return np.maximum(0, z)

    def ReLU_der(self, z):
        #return np.where(z > 0, 1.0, 0.0)
        return elementwise_grad(self.ReLU)(z)

    def Leaky_ReLU(self, z, alpha = 0.1):
        return np.where(z > 0, z, alpha * z)

    def Leaky_ReLU_der(self, z, alpha = 0.1):
        return elementwise_grad(self.Leaky_ReLU)(z, alpha = alpha)

    def soft_max(self, z):
        #z = np.where(value >  500,  500, z)
        #z = np.where(value < -500, -500, z)
        return np.exp(z) / (np.sum(np.exp(z), axis=1, keepdims=True))

    def soft_max_der(self, z):
        #return self.soft_max(x) * (1 - self.soft_max(x))
        return elementwise_grad(self.soft_max)(z)
    
    #################################### ACTIVATION ####################################

    #################################### ACCURACY ####################################
    def accuracy_score(self, X, target):
        if self.allowed_acc:
            return np.sum(np.around(self.predict(X)) == target, axis=0) / target.shape[0]
        else:
            raise ValueError(f"accuracy_score can only be performed on classification datasets")

    def MSE_score(self, X, target):
        return np.mean((self.predict(X) - target)**2)

    def R2_score(self, X, target):
        return 1 - np.sum((target - self.predict(X))**2) / np.sum((target - np.mean(target, axis=0) ** 2))
    
    #################################### ACCURACY ####################################

    #################################### COST ####################################
    def MSE(self, y_tilde):
        return (y_tilde - self.y)**2

    def MSE_der(self, y_tilde):
        #return - (2 * y_tilde) + (2 * self.y)
        return elementwise_grad(self.MSE)(y_tilde)

    def cross_entropy(self, y_tilde):
        return -(self.y * np.log(y_tilde) + (1 - self.y) * np.log(1 - y_tilde))

    def cross_entropy_der(self, y_tilde):
        return elementwise_grad(self.cross_entropy)(y_tilde)
    
    #################################### COST ####################################
