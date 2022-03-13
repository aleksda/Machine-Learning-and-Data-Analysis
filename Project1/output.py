class PrintError:
    def __init__(self, scaler_obj):
        self.scaler = scaler_obj
        self.ManualDict = None
        self.ScikitDict = None
        self.ErrorDict = None
        
        
        
    def ManualError(self, beta, printToScreen : bool = False):
        import error as err
        
        X_train_scaled = self.scaler.X_train_scaled; Z_train = self.scaler.Z_train
        X_test_scaled  = self.scaler.X_test_scaled; Z_test = self.scaler.Z_test
        
        print(self.ManualError.__name__)
        if type(beta).__module__ == "numpy":
            print("beta: ", beta.__class__.__module__)
            Z_tilde   = X_train_scaled @ beta
            Z_pred    = X_test_scaled @ beta
        else:
            print("beta: ", beta.__class__.__module__, beta.__class__.__name__)
            Z_tilde = beta.predict(X_train_scaled)
            Z_pred  = beta.predict(X_test_scaled)
        
        # Error analysis
        Err_tilde = err.Error(Z_tilde)
        
        MSE_tilde = Err_tilde.MSE(Z_train, Z_tilde)
        MAE_tilde = Err_tilde.MAE(Z_train, Z_tilde)
        R2_tilde  = Err_tilde.R2_Score(Z_train, Z_tilde)
        
        Err_pred  = err.Error(Z_pred)
        
        MSE_pred = Err_pred.MSE(Z_test, Z_pred)
        MAE_pred = Err_pred.MAE(Z_test, Z_pred)
        R2_pred  = Err_pred.R2_Score(Z_test, Z_pred)
        
        self.ErrorDict = {"R2 Score" : (R2_tilde, R2_pred),\
                           "MSE" : (MSE_tilde, MSE_pred), "MAE" : (MAE_tilde, MAE_pred)}
        # self.ManualDict = {"R2 Score" : (R2_tilde, R2_pred),\
                           # "MSE" : (MSE_tilde, MSE_pred), "MAE" : (MAE_tilde, MAE_pred)}
            
        if printToScreen:
            self.printout(self.ManualError)
        
        
    def ScikitError(self, beta, printToScreen : bool = False):
        from sklearn.metrics import mean_squared_error, r2_score,\
            mean_squared_log_error, mean_absolute_error
        
        X_train_scaled = self.scaler.X_train_scaled; Z_train = self.scaler.Z_train
        X_test_scaled  = self.scaler.X_test_scaled; Z_test = self.scaler.Z_test
        
        print(self.ScikitError.__name__)
        if type(beta).__module__ == "numpy":
            print("beta: ", beta.__class__.__module__)
            Z_tilde   = X_train_scaled @ beta
            Z_pred    = X_test_scaled @ beta
        else:
            print("beta: ", beta.__class__.__module__, beta.__class__.__name__)
            Z_tilde = beta.predict(X_train_scaled)
            Z_pred  = beta.predict(X_test_scaled)
        
        # print('The intercept alpha: \n', beta.intercept_.shape)
        # print('Coefficient beta : \n', beta.coef_.shape)
        
        # Z_tilde = scikit_obj.predict(X_train_scaled)
        # Z_pred  = scikit_obj.predict(X_test_scaled)
        
        # print(Z_regfit[0:4, :], "\n...", Z_tilde[0:4, :])
        
        # Error analysis
        MSE_tilde = mean_squared_error(Z_train, Z_tilde)
        MAE_tilde = mean_absolute_error(Z_train, Z_tilde)
        R2_tilde = r2_score(Z_train, Z_tilde)
        
        MSE_pred = mean_squared_error(Z_test, Z_pred)
        MAE_pred = mean_absolute_error(Z_test, Z_pred)
        R2_pred  = r2_score(Z_test, Z_pred)
        
        self.ErrorDict = {"R2 Score" : (R2_tilde, R2_pred),\
                           "MSE" : (MSE_tilde, MSE_pred), "MAE" : (MAE_tilde, MAE_pred)}
        # self.ScikitDict = {"R2 Score" : (R2_tilde, R2_pred),\
                           # "MSE" : (MSE_tilde, MSE_pred), "MAE" : (MAE_tilde, MAE_pred)}
        
        if printToScreen:
            self.printout(self.ScikitError)
        
    def printout(self, class_method):
        ErrorDict = self.ErrorDict
        
        print("-----------Tilde-----------")
        print("R2 Score:", ErrorDict["R2 Score"][0],\
              "MSE:", ErrorDict["MSE"][0], "MAE:", ErrorDict["MAE"][0])
        print("---------------------------\n")
        
        print("-----------Predict----------")
        print("R2 Score:", ErrorDict["R2 Score"][1],\
              "MSE:", ErrorDict["MSE"][1], "MAE:", ErrorDict["MAE"][1])
        print("----------------------------\n")
        """
        if class_method.__name__ == self.ManualError.__name__:
            
            ManualDict = self.ManualDict
            
            print("-----------Tilde-----------")
            print("R2 Score(Wrong):", ManualDict["R2 Score"][0],\
                  "MSE:", ManualDict["MSE"][0], "MAE:", ManualDict["MAE"][0])
            print("---------------------------\n")
            
            print("-----------Predict----------")
            print("R2 Score(Wrong):", ManualDict["R2 Score"][1],\
                  "MSE:", ManualDict["MSE"][1], "MAE:", ManualDict["MAE"][1])
            print("----------------------------\n")
            
        elif class_method.__name__ == self.ScikitError.__name__:
            
            ScikitDict = self.ScikitDict
            
            print("-----------Tilde-----------")
            print("R2 Score:", ScikitDict["R2 Score"][0],\
                  "MSE:", ScikitDict["MSE"][0], "MAE:", ScikitDict["MAE"][0])
            print("---------------------------\n")
            
            print("-----------Predict----------")
            print("R2 Score:", ScikitDict["R2 Score"][1],\
                  "MSE:", ScikitDict["MSE"][1], "MAE:", ScikitDict["MAE"][1])
            print("----------------------------\n")
        """