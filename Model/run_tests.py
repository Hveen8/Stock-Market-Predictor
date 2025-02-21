import numpy as np
import pandas as pd
from src.data_preprocessor import DataPreprocessor
from src.lstm_model import LSTMModel
from src.forecast_engine import ForecastEngine
from src.visualizer import Visualizer
import matplotlib.pyplot as plt

# ------------------------------
#          Run Script
# ------------------------------
def run():
    # 1. Load and Prepare Data
    data_dir = '/mnt/slurm_nfs/ece498_w25_20/Stock-Market-Predictor/Model/data/'
    # Ensure the CSV is divided into columns named 'System1', 'System2', etc.
    file = 'UTD_Load_sorted.csv'
    try:
        df = pd.read_csv(data_dir+file)
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
        lstm_model = LSTMModel(layers=2,
                                isReturnSeq=False,
                                look_back=look_back,
                                batch_size=batch_size,
                                neurons=neurons,
                                epochs=epochs,
                                activation=activation,
                                dropout=dropout)
        
        lstm_model.train(trainX, trainY)
        
        # *** This MUST be called
        trainPredict = lstm_model.predict(trainX)

        # 5. Forecast Future Values        
        # Create ForecastEngine instance with parameters from lstm_model or defaults.
        forecast_engine = ForecastEngine(trained_model=lstm_model,  # Use trained model's layers if None
                                          isReturnSeq=True)  # Force return sequences to True for forecasting

        # start_input = trainX[-1].reshape(1, look_back, 1)
        # start_input = trainX

        forecast_steps = 2000
        # *** This MUST be called           
        future_predictions = forecast_engine.forecast(trainX, forecast_steps)

        # 6. Visualize Results
        visualizer = Visualizer(scaler=data_preprocessor.scaler,
                                trained_model=lstm_model,
                                forecast_engine=forecast_engine)
        visualizer.plot_results(curr_dataset, curr_system, curr_dir)

if __name__ == "__main__":
    run()