import time
import math
import traceback
import numpy as np
import pandas as pd
import tensorflow as tf
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

    # 1. Load and Prepare Data
    # Ensure the CSV is divided into columns named 'System1', 'System2', etc.
    # file = 'UTD_Load_sorted.csv'
    file = 'Stock Data.csv'
    try:
        df = pd.read_csv(data_dir+file)
    except FileNotFoundError:
        print(f"Error: '{file}' not found. Place it into the '/data' directory")
        return

    # # Parameters
    # look_back = 6000 # bet look back ratio 0.6:1 >> 10000 -> ~6000, need to round to just whole 1000th
    # batch_size = 128
    # neurons = 100
    # epochs = 10
    # headroom = 1.0
    # is2Layer = True
    # activation = 'tanh'  # or 'relu' but relu is shit
    # dropout = 0.1

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
    batch_size_list = [256]
    # look_back_list  = list(range(1000, 4001, 250))
    look_back_list = [4000]
    # epoch_list      = list(range(2, 41, 1))
    epoch_list      = [18]
    # headroom_list   = [1.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    headroom_list   = [1.0]
    dropout_list    = [0]
    # dropout_list    = np.linspace(1.0, 0.0, 11)
    activation      = 'tanh'
    layers          = 2
    neurons         = 100
    # forecast_steps  = 2000


    # Iterate through each column (assuming each column represents a system)
    stocks = ['AAPL', 'NVDA']
    for stock in stocks:

        if not stock == 'AAPL':
            continue

        # Getting only row for stock
        # df_indexes = df[df['symbol'] == stock].copy()

        # curr_dataset = df[curr_system].values.reshape(-1, 1).astype('float32')

        curr_dataset = filter_multi_features(df, stock, feature_cols)

        curr_dir = 'results10'

        # ======================================================================== #
        forecast_horizon = 100   # number of points to forecast per fold
        initial_train_size = 4500  # choose your training size
        step_size = 0              # roll the window forward by this many points
        # ======================================================================== #

        optimal_model_params = None
        cross_val_times = None
        lowest_rmse = 1000000
        for bs in batch_size_list:
            for lb in look_back_list:
                for ep in epoch_list:
                    for hm in headroom_list:
                        for do in dropout_list:
                            try:
                                look_back = lb
                                batch_size = bs
                                epochs = ep
                                headroom = hm
                                dropout = do

                                # look_back = math.ceil(initial_train_size*0.6)

                                model_params = {'features': 9,
                                                'look_back': look_back,
                                                'batch_size': batch_size,
                                                'epochs': epochs,
                                                'headroom': headroom,
                                                'dropout': dropout,
                                                'layers': layers,
                                                'neurons': neurons,
                                                'activation': activation}

                                # # alpha_range = np.arange(0.1, 1.0, 0.1)
                                # alpha_range = [0.1]
                                # # beta_range = np.arange(0.1, 1.0, 0.1)
                                # # beta_range = [0.4, 0.5, 0.6]
                                # beta_range = [0.5]
                                # # weight_range = np.arange(0.005, 0.1, 0.005)
                                # weight_range = [0.02, 0.025, 0.03, 0.035]

                                # # Generate all possible (alpha, beta, weight) combinations
                                # taf_params_list = [(round(alpha, 2), round(beta, 2), round(weight, 3)) 
                                #                     for alpha in alpha_range 
                                #                     for beta in beta_range 
                                #                     for weight in weight_range]

                                # print("Generated TAF parameter combinations:")
                                # print(taf_params_list)

                                
                                model, train_data_inverted, train_end, test_end, non_taf_forecast, local_rmse = time_series_cross_validation(curr_dataset, model_params, forecast_horizon, initial_train_size, step_size, target_feature_col, False)

                                print(f'RMSE: {local_rmse} | Look_back: {look_back}, Epochs: {epochs}')
                                
                                if local_rmse < lowest_rmse:
                                    lowest_rmse = local_rmse
                                    optimal_model_params = model_params
                                    cross_val_times = {'model': model,
                                                       'train_data_inverted': train_data_inverted,
                                                       'train_end': train_end,
                                                       'test_end': test_end,
                                                       'non_taf_forecast': non_taf_forecast,
                                                       'local_rmse': local_rmse}
                            
 

                                # print("Cross-Validation RMSEs:", rmse_list)
                                # print("Mean RMSE:", np.mean(rmse_list))

                                # # 2. Data Preprocessing
                                # data_preprocessor = DataPreprocessor(headroom=headroom)
                                # scaled_data = data_preprocessor.fit_transform(curr_dataset)

                                # print(f'curr_dataset shape: ', curr_dataset.shape)
                                # print(f'scaled_data shape: ', scaled_data.shape)

                                # # 3. Create Dataset for LSTM
                                # dataX, dataY = data_preprocessor.create_dataset(scaled_data, look_back=look_back)
                                # dataX, dataY = data_preprocessor.trim_XY(dataX, dataY, batch_size)

                                # # Reshape input to be [samples, time steps, features] which is required for LSTM
                                # # *****Need to tie input layer to Model class*****
                                # trainX = np.reshape(dataX, (dataX.shape[0], dataX.shape[1], 1))
                                # print('trainX shape (After Reshape): ', trainX.shape)
                                # trainY = dataY

                                # # 4. Train LSTM Model
                                # lstm_model = LSTMModel(layers=layers,
                                #                         isReturnSeq=False,
                                #                         look_back=look_back,
                                #                         batch_size=batch_size,
                                #                         neurons=neurons,
                                #                         epochs=epochs,
                                #                         activation=activation,
                                #                         dropout=dropout)
                                
                                # lstm_model.train(trainX, trainY)
                                
                                # # *** This MUST be called
                                # trainPredict = lstm_model.predict(trainX)
                                # print('Training infered data shape: ', trainPredict.shape)

                                # # 5. Forecast Future Values        
                                # # Create ForecastEngine instance with parameters from lstm_model or defaults.
                                # forecast_engine = ForecastEngine(trained_model=lstm_model,
                                #                                 isReturnSeq=True)  # Force return sequences to True for forecasting

                                # # start_input = trainX[-1].reshape(1, look_back, 1)
                                # # start_input = trainX

                                # forecast_steps = 2000
                                # # *** This MUST be called           
                                # futurePredictions = forecast_engine.forecast(trainX, forecast_steps)
                                # print('Forecast infered data shape: ', futurePredictions.shape)

                                # # 6. Visualize Results
                                # visualizer = Visualizer(scaler=data_preprocessor.scaler,
                                #                         trained_model=lstm_model,
                                #                         forecast_engine=forecast_engine)
                                
                                # visualizer.plot_results(curr_dataset, trainY, curr_system, results_dir+curr_dir)
                            except Exception as e:
                                print("+======================================================================================+")
                                print("+======================================================================================+")
                                print("ERROR! Details: ")
                                traceback.print_exc()
                                print("+======================================================================================+")
                                print("+======================================================================================+")
                                continue
        

        visualizer = Visualizer(scaler=cross_val_times['model'][0].scaler,
                                trained_model=cross_val_times['model'][1],
                                forecast_engine=cross_val_times['model'][2])        
        visualizer.plot_results(cross_val_times['local_rmse'], cross_val_times['train_data_inverted'], cross_val_times['train_end'], cross_val_times['test_end'], cross_val_times['non_taf_forecast'], curr_dataset, stock, target_feature_col, results_dir+curr_dir, [0, 0, 0])
        print("Cross-Validation RMSEs:", cross_val_times['local_rmse'])   


        # == Doing it agian for TAF, not the best ==

        model, train_data_inverted, train_end, test_end, rmse_TAFs = time_series_cross_validation(curr_dataset, optimal_model_params, forecast_horizon, initial_train_size, step_size, target_feature_col, True)

        # model, train_data_inverted, train_end, test_end, forecasted_inverted, rmse_list = time_series_cross_validation(curr_dataset, model_params, forecast_horizon, initial_train_size, step_size)

        visualizer = Visualizer(scaler=model[0].scaler,
                                trained_model=model[1],
                                forecast_engine=model[2])
        # visualizer.plot_results(np.mean(rmse_list), train_data_inverted, train_end, test_end, forecasted_inverted, curr_dataset, curr_system, results_dir+curr_dir)
        for (alpha, beta, weight), (rmse_taf, adjusted_forecast) in rmse_TAFs.items():
            visualizer.plot_results(rmse_taf, train_data_inverted, train_end, test_end, adjusted_forecast, curr_dataset, stock, target_feature_col, results_dir+curr_dir, [alpha, beta, weight])
            print("Cross-Validation RMSEs (TAF):", rmse_taf)   

if __name__ == "__main__":
    run()