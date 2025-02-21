import numpy as np
import pandas as pd
from data_preprocessor import DataPreprocessor
from lstm_model import LSTMModel
from forecast_engine import ForecastEngine
from visualizer import Visualizer
import matplotlib.pyplot as plt

# ------------------------------
#          Run Script
# ------------------------------
def run():
    # 1. Load and Prepare Data
    dir = '/mnt/slurm_nfs/ece498_w25_20/Stock-Market-Predictor/Model/data'
    # Ensure the CSV is divided into columns named 'System1', 'System2', etc.
    file = 'UTD_Load_sorted.csv'
    try:
        df = pd.read_csv('your_data.csv')
    except FileNotFoundError:
        print(f"Error: '{file}' not found. Place it into the '/data' directory")
        return

    # Parameters
    look_back = 6000 # bet look back ratio 0.6:1 >> 10000 -> ~6000, need to round to just whole 1000th
    batch_size = 128
    neurons = 50
    epochs = 10
    headroom = 1.0
    is2Layer = True
    activation = 'tanh'  # or 'relu' but relu is shit
    dropout = 0.0

    # Iterate through each column (assuming each column represents a system)
    for curr_system in df.columns:
        curr_dataset = df[curr_system].values.reshape(-1, 1)

        curr_dir = 'results1'

        # 2. Data Preprocessing
        data_preprocessor = DataPreprocessor(headroom=headroom)
        scaled_data = data_preprocessor.fit_transform(curr_dataset)

        # 3. Create Dataset for LSTM
        dataX, dataY = data_preprocessor.create_dataset(scaled_data, look_back=look_back)
        dataX, dataY = data_preprocessor.trim_XY(dataX, dataY, batch_size)

        # Reshape input to be [samples, time steps, features] which is required for LSTM
        # *Need to tie input layer to Model class*
        trainX = np.reshape(dataX, (dataX.shape[0], dataX.shape[1], 1))
        trainY = dataY

        # 4. Train LSTM Model
        lstm_model = LSTMModel(look_back=look_back,
                                batch_size=batch_size,
                                neurons=neurons,
                                epochs=epochs,
                                is2Layer=is2Layer,
                                activation=activation,
                                dropout=dropout)
        lstm_model._build_model()
        lstm_model.train(trainX, trainY)
        # *** This MUST be called
        trainPredict = lstm_model.predict(trainX)

        # 5. Forecast Future Values
        # The length of the start_input sequence must match batch_size parameter
        #start_input = scaled_data[-look_back:].reshape(1, look_back, 1)
        start_input = trainX.reshape(1, look_back, 1)
        # Number of future steps to forecast
        forecast_steps = 2000
        forecast_engine = ForecastEngine(trained_model=lstm_model.model,
                                            look_back=look_back,
                                            batch_size=batch_size,
                                            neurons=neurons,
                                            is2Layer=is2Layer)
        # *** This MUST be called           
        future_predictions = forecast_engine.forecast(start_input, forecast_steps)

        # 6. Visualize Results
        visualizer = Visualizer(scaler=data_preprocessor.scaler,
                                trained_model=lstm_model,
                                forecast_engine=forecast_engine)
        visualizer.plot_results(curr_dataset, curr_system, curr_dir)

if __name__ == "__main__":
    main()