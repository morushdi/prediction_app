import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import time
import os
from math import sqrt
import pickle
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error, max_error

# define linear regression model class with custom functions

class LR():
    def __init__(self, model_path):
        self.model_path = model_path
        self.model  = LinearRegression()
    
    def train(self, X_train, y_train):
        t_start = time.process_time()
        self.model.fit(X_train, y_train)
        t_end = time.process_time()

        self.save_model()

        return {'train_time': t_end - t_start}
    
    def evaluate(self, X_test, Y_test):
        predictions = self.model.predict(X_test)
        metrics = {'predictions': predictions,
                   'MSE': mean_squared_error(Y_test, predictions),
                   'RMSE': sqrt(mean_squared_error(Y_test, predictions)),
                   'C_D': r2_score(Y_test, predictions),
                   'Exp_Var': explained_variance_score(Y_test, predictions, multioutput='uniform_average'),
                   'MAE': mean_absolute_error(Y_test, predictions)}
        
        return metrics
    
    def predict(self, X):
        if os.path.isfile(self.model_path):
            self.model = pickle.load(open(self.model_path, 'rb'))
        else:
            raise Exception(f'Model not found in {self.model_path}')
        return self.model.predict(X)
    

    def save_model(self):
        pickle.dump(self.model, open(self.model_path, 'wb'))


if __name__ == '__main__':
    print("Running main")
    # --- load data --- #
    df = pd.read_csv('data/df.csv')
    df = df.dropna()
    # --- Task 1 --- #
    lr = LR('models/lr-single.pkl')
    features = ["Electrical power", "Yaw misalignment", "Mean pitch angle", "Rotor azimuth angle", "Generator speed", 
                "Blade root 1 Mx", "Blade root 1 My", "Blade root 2 Mx", "Blade root 2 My", "Blade root 3 Mx", "Blade root 3 My", "previous Hub wind speed magnitude"]
    x = pd.DataFrame(df, columns=features)
    y = df['Hub wind speed magnitude']
    print("training single linear regression model")
    lr.train(x, y)
    print("Saved")
    
    # --- Task 2 --- #
    lr_multi = LR('models/lr-multi.pkl')
    features = ["Electrical power", "Yaw misalignment", "Mean pitch angle", "Rotor azimuth angle", "Generator speed", 
            "Blade root 1 Mx", "Blade root 1 My", "Blade root 2 Mx", "Blade root 2 My", "Blade root 3 Mx", "Blade root 3 My", 
            "Nacelle roll acceleration", "Nacelle yaw acceleration" , "Nacelle nod acceleration",
            "MXT(Tower Bottom)", "MYT(Tower Bottom)", "MZT(Tower Bottom)", 
            "MXT(Tower Top)", "MYT(Tower Top)", "MZT(Tower Top)"]

    targets = ["Stationary hub Mx", "Stationary hub My", "Stationary hub Mz", "Stationary hub Fx"]
    x = pd.DataFrame(df, columns=features)
    y = pd.DataFrame(df, columns=targets)
    print("training multi linear regression model")
    lr_multi.train(x, y)
    print("Saved")
