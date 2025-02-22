import time
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from src.data_preprocessor import DataPreprocessor
from src.lstm_model import LSTMModel
from src.forecast_engine import ForecastEngine
from src.visualizer import Visualizer
from src.utils import calculate_rmse
from src.cross_validation import time_series_cross_validation
import matplotlib.pyplot as plt

# Enabling multi-GPU useage on 1 node
# gpu_strategy = tf.distribute.MirroredStrategy()
#print(f"Number of GPUs Available: {gpu_strategy.num_replicas_in_sync}")

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
    file = 'UTD_Load_sorted.csv'
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

    batch_size_list = [128, 256]
    # batch_size_list = [256]
    # look_back_list  = [5000, 6000]
    look_back_list  = [6000]
    epoch_list      = [1, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    # headroom_list   = [1.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    headroom_list   = [1.0]
    dropout_list    = [0]
    activation      = 'tanh'
    layers          = 2
    neurons         = 100
    forecast_steps  = 2000


    # Iterate through each column (assuming each column represents a system)
    for curr_system in df.columns:

        if curr_system != 'B1':
            continue

        curr_dataset = df[curr_system].values.reshape(-1, 1).astype('float32')

        curr_dir = 'results1'

        for bs in batch_size_list:
            for lb in look_back_list:
                for ep in epoch_list:
                    for hm in headroom_list:
                        for do in dropout_list:
                            # try:
                            # look_back = lb
                            batch_size = bs
                            epochs = ep
                            headroom = hm
                            dropout = do

                            forecast_horizon = 4000    # number of points to forecast per fold
                            initial_train_size = 8000  # choose your training size
                            step_size = 0           # roll the window forward by this many points

                            look_back = math.ceil(initial_train_size*0.6)

                            model_params = {'look_back': look_back,
                                            'batch_size': batch_size,
                                            'epochs': epochs,
                                            'headroom': headroom,
                                            'dropout': dropout,
                                            'layers': layers,
                                            'neurons': neurons,
                                            'activation': activation}

                            model, train_data_inverted, forecasted_inverted, rmse_list = time_series_cross_validation(curr_dataset, model_params, forecast_horizon, initial_train_size, step_size)

                            visualizer = Visualizer(scaler=model[0].scaler,
                                                    trained_model=model[1],
                                                    forecast_engine=model[2])
                            visualizer.plot_results(train_data_inverted, forecasted_inverted, curr_dataset, curr_system, results_dir+curr_dir)

                            print("Cross-Validation RMSEs:", rmse_list)
                            print("Mean RMSE:", np.mean(rmse_list))

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
                            # except Exception as e:
                            #     print("ERROR! Details:", e)
                            #     continue

if __name__ == "__main__":
    run()