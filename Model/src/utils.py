import numpy as np
from sklearn.metrics import mean_squared_error

def calculate_rmse(true_values, predicted_values):
    return np.sqrt(mean_squared_error(true_values, predicted_values))