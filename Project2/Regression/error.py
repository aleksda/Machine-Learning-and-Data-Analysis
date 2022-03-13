import numpy as np
import time

class Error:
    def __init__(self, data):
        """
        Takes in the data to find it's length(1D)/size(multi-D).
        Used for error analysis.
        Use either raw or model data; assuming same dimensions on both.
        
        Parameters
        ----------
        data : Data to be analyzed.

        Returns
        -------
        None.

        """
        
        self.N = data.shape[0]
            
        
    def MSE(self, y_data, y_model):
        """ MSE Error """
        return np.sum((y_data - y_model)**2)/self.N
        
    def MAE(self, y_data, y_model):
        """ MAE Error """
        return np.sum(abs(y_data - y_model))/self.N
    
    def R2_Score(self, y_data, y_model):
        """ R2_Score """
        
        # error[degree] = np.mean( np.mean((y_test - y_pred)**2, axis=1, keepdims=True) )
        # ymean = np.mean(y_data, axis = 0)
        # return 1 - np.sum((y_data - y_model)**2)/np.sum((y_data - ymean)**2)
        
        # ymean = np.mean(y_data, axis = 1)
        # diff = (y_data.transpose() - ymean).transpose()
        # return 1 - np.sum((y_data - y_model)**2)/np.sum(diff**2)
        
        return 1. - np.sum((y_data - y_model)**2)/np.sum((y_data - np.mean(y_data))**2)
        # return 1 - np.sum((y_data[:-2] - y_model[:-2])**2)/np.sum((y_data[:-2] - np.mean(y_data))**2)
        
    def Accuracy(self, y_data, y_model):
        # print("ACcuracy yay!")
        sum_correct = np.sum(y_data == y_model)
        return sum_correct/self.N