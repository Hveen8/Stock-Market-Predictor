import math
import numpy as np
from src.data_preprocessor import DataPreprocessor
from src.lstm_model import LSTMModel
from src.forecast_engine import ForecastEngine
from src.utils import calculate_rmse

def time_series_cross_validation(curr_dataset, model_params, forecast_horizon, initial_train_size, step_size):
    """
    curr_dataset: the full dataset (2D array, e.g. shape (n_samples, 1))
    model_params: dict with keys: look_back, batch_size, epochs, headroom, dropout, etc.
    forecast_horizon: number of points to forecast in each fold
    initial_train_size: the initial number of samples used for training
    step_size: number of samples to roll forward between folds

    Returns: list of RMSE values, one per fold
    """
    rmse_list = []
    n = len(curr_dataset)
    start = 0
    # for start in range(0, n - initial_train_size - forecast_horizon + 1, step_size): 
    train_data = curr_dataset[start:start+initial_train_size]

    data_preprocessor = DataPreprocessor(headroom=model_params['headroom'])
    scaled_train = data_preprocessor.fit_transform(train_data)

    dataX, dataY = data_preprocessor.create_dataset(scaled_train, look_back=model_params['look_back'])
    dataX, dataY = data_preprocessor.trim_XY(dataX, dataY, model_params['batch_size'])

    effective_train_samples = len(dataX)
    effective_train_end = start + effective_train_samples + model_params['look_back']
    test_end = effective_train_end + math.ceil(forecast_horizon/model_params['batch_size'])*model_params['batch_size']
    test_data = curr_dataset[effective_train_end:test_end]

    print(f"Fold (Cross Validation) with train indices {start}:{effective_train_end} and test indices {effective_train_end}:{test_end}")

    # *****Need to tie input layer to Model class*****
    trainX = np.reshape(dataX, (dataX.shape[0], dataX.shape[1], 1))
    trainY = dataY

    lstm_model = LSTMModel(layers=model_params['layers'],
                            isReturnSeq=False,
                            look_back=model_params['look_back'],
                            batch_size=model_params['batch_size'],
                            neurons=model_params['neurons'],
                            epochs=model_params['epochs'],
                            activation=model_params['activation'],
                            dropout=model_params['dropout'])
    lstm_model.train(trainX, trainY)
    trainPredict = lstm_model.predict(trainX)

    forecast_engine = ForecastEngine(trained_model=lstm_model, isReturnSeq=True)
    forecastPredict = forecast_engine.forecast(trainX, forecast_horizon)
    
    # Invert the scaling for the forecast, train and test data
    forecasted_inverted = data_preprocessor.scaler.inverse_transform(forecastPredict)
    train_data_inverted = data_preprocessor.scaler.inverse_transform(trainPredict)
    # test_data_inverted  = data_preprocessor.scaler.inverse_transform(test_data)
    
    # Calculate RMSE between forecast and actual test data
    # rmse = np.sqrt(np.mean((forecasted_inverted[:, 0] - test_data_inverted[:, 0]) ** 2))
    rmse = calculate_rmse(forecasted_inverted[:, 0], test_data[:, 0])
    print(f"Fold RMSE: {rmse:.2f}")
    rmse_list.append(rmse)

    return [data_preprocessor, lstm_model, forecast_engine], train_data_inverted, effective_train_end, test_end, forecasted_inverted, rmse_list