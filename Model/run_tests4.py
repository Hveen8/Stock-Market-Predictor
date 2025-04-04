import time
import math
import traceback
import numpy as np
import pandas as pd
import tensorflow as tf
from bayes_opt import BayesianOptimization

from src.data_preprocessor import DataPreprocessor, filter_multi_features
from src.lstm_model import LSTMModel
from src.forecast_engine import ForecastEngine
from src.visualizer import Visualizer
from src.utils import calculate_rmse
from src.cross_validation import time_series_cross_validation
import matplotlib.pyplot as plt

# Enabling multi-GPU useage on 1 node
# gpu_strategy = tf.distribute.MirroredStrategy()
# print(f"Number of GPUs Available: {gpu_strategy.num_replicas_in_sync}")

gpus = len(tf.config.list_physical_devices('GPU'))
print(f"Num GPUs Available: {gpus}")
#print(f"Worker (1 task per node) {os.environ.get('SLURM_PROCID', 'N/A')} sees {len(gpus)} GPU(s).")

# Tensorflow distributed compute strategy (some shit that makes a distributed environment/rule-set)
# gpu_strategy = tf.distribute.MultiWorkerMirroredStrategy()

# ------------------------------
#          Run Script
# ------------------------------
def run():
    
    data_dir = '/mnt/slurm_nfs/ece498_w25_20/Stock-Market-Predictor/Model/data/'
    results_dir = '/mnt/slurm_nfs/ece498_w25_20/Stock-Market-Predictor/Model/results/'

    curr_dir = 'results12'

    # 1. Load and Prepare Data
    # Ensure the CSV is divided into columns named 'System1', 'System2', etc.
    # file = 'UTD_Load_sorted.csv'
    file = 'Stock Data.csv'
    try:
        df = pd.read_csv(data_dir+file)
    except FileNotFoundError:
        print(f"Error: '{file}' not found. Place it into the '/data' directory")
        return

# ==================== Global Parameters ====================
    feature_cols = ['open',
                    'high',
                    'low',
                    'close',
                    'volume',
                    'rsi',
                    'macd',
                    'macdh',
                    'macds']
    target_feature_col = 3
    features = len(feature_cols)
    
    # Fixed parameters
    batch_size = 1024
    headroom = 2.0
    dropout = 0.0
    layers = 2
    neurons = 100
    activation = 'tanh'
    # activation = 'relu' # NO Good!
    
    # Forecasting and dataset parameters
    forecast_horizon = 60   # Number of future points to forecast per fold
    initial_train_size = 4500
    step_size = 0            # For rolling window (0 -> no rolling)

    stocks = ['AAPL', 'NVDA']
    

    for stock in stocks:
        if stock != 'AAPL':
            continue

        curr_dataset = filter_multi_features(df, stock, feature_cols)

        # ==================== Bayesian Optimization Setup ====================
        # Define the objective function for Bayesian Optimization
        def objective(look_back, epochs):
            look_back = int(look_back)
            epochs = int(epochs)
            
            model_params = {
                'features': features,
                'look_back': look_back,
                'batch_size': batch_size,
                'epochs': epochs,
                'headroom': headroom,
                'dropout': dropout,
                'layers': layers,
                'neurons': neurons,
                'activation': activation
            }
            
            try:
                # Run cross-validation (non-TAF version) and obtain RMSE.
                # time_series_cross_validation (FALSE) is expected to return:
                # model_components, train_predict_inverted, effective_train_end, test_end, forecasted_inverted, rmse_taf_preTAF
                _, _, _, _, _, rmse = time_series_cross_validation(
                    curr_dataset, model_params, forecast_horizon, initial_train_size, step_size, target_feature_col, False)
            except Exception as e:
                print("Error during evaluation:", e)
                traceback.print_exc()
                rmse = 1e6

            # BayesianOptimization maximizes the objective so return negative RMSE.
            return -rmse

        # *****************************
        # Define parameter bounds
        pbounds = {
            'look_back': (1467, 1467),
            'epochs': (78, 78)
        }
        # *****************************
        
        optimizer = BayesianOptimization(
            f=objective,
            pbounds=pbounds,
            random_state=42  # For reproducibility
        )
        
        print("Starting Bayesian optimization for:", stock)
        optimizer.maximize(
            init_points=5,   # Number of random initialization points
            n_iter=20        # Number of iterations for the optimization
        )
        
        best_params = optimizer.max['params']
        optimal_look_back = int(best_params['look_back'])
        optimal_epochs = int(best_params['epochs'])
        print(f"Optimal parameters for {stock}: look_back={optimal_look_back}, epochs={optimal_epochs}")
        
        # ==================== Run Cross-Validation with Optimal Parameters ====================
        optimal_model_params = {
            'features': features,
            'look_back': optimal_look_back,
            'batch_size': batch_size,
            'epochs': optimal_epochs,
            'headroom': headroom,
            'dropout': dropout,
            'layers': layers,
            'neurons': neurons,
            'activation': activation
        }
        
        model, train_data_inverted, train_end, test_end, non_taf_forecast, rmse_non_taf, rmse_TAFs = time_series_cross_validation(curr_dataset, optimal_model_params, forecast_horizon, initial_train_size, step_size, target_feature_col, True)
        
        visualizer = Visualizer(scaler=model[0].scaler,
                                trained_model=model[1],
                                forecast_engine=model[2])
        visualizer.plot_results(rmse_non_taf, train_data_inverted, train_end, test_end, non_taf_forecast, curr_dataset, stock, target_feature_col, results_dir+curr_dir, [0, 0, 0])
        print("|=====================================|")
        print("Cross-Validation RMSEs (Non-TAF):", rmse_non_taf)
        print("|=====================================|")
        for (alpha, beta, weight), (rmse_taf, adjusted_forecast) in rmse_TAFs.items():
            visualizer.plot_results(rmse_taf, train_data_inverted, train_end, test_end, adjusted_forecast, curr_dataset, stock, target_feature_col, results_dir+curr_dir, [alpha, beta, weight])
            print("|=====================================|")
            print("Cross-Validation RMSEs (TAF):", rmse_taf)
            print("|=====================================|")

if __name__ == "__main__":
    run()