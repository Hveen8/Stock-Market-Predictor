import numpy as np
from src.data_preprocessor import DataPreprocessor
from src.lstm_model import LSTMModel
from src.forecast_engine import ForecastEngine

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
    
    # Walk-forward loop: ensure we have enough data for training + forecast in each fold
    for start in range(0, n - initial_train_size - forecast_horizon + 1, step_size):
        train_end = start + initial_train_size  # end of training data for current fold
        test_end = train_end + forecast_horizon   # end of test data for current fold

        train_data = curr_dataset[start:train_end]
        test_data  = curr_dataset[train_end:test_end]

        print(f"Fold with train indices {start}:{train_end} and test indices {train_end}:{test_end}")

        # Preprocess training data using your DataPreprocessor (using provided headroom)
        data_preprocessor = DataPreprocessor(headroom=model_params['headroom'])
        scaled_train = data_preprocessor.fit_transform(train_data)

        # Create training dataset for LSTM
        dataX, dataY = data_preprocessor.create_dataset(scaled_train, look_back=model_params['look_back'])
        dataX, dataY = data_preprocessor.trim_XY(dataX, dataY, model_params['batch_size'])
        # *****Need to tie input layer to Model class*****
        trainX = np.reshape(dataX, (dataX.shape[0], dataX.shape[1], 1))
        trainY = dataY

        # Initialize and train model on current fold’s training data
        lstm_model = LSTMModel(
            layers=model_params.get('layers'),
            isReturnSeq=False,
            look_back=model_params['look_back'],
            batch_size=model_params['batch_size'],
            neurons=model_params['neurons'],
            epochs=model_params['epochs'],
            activation=model_params['activation'],
            dropout=model_params['dropout']
        )
        lstm_model.train(trainX, trainY)
        trainPredict = lstm_model.predict(trainX)

        # Forecast for the horizon: note that here you use the forecast engine
        forecast_engine = ForecastEngine(trained_model=lstm_model, isReturnSeq=True)
        forecasted = forecast_engine.forecast(trainX, forecast_horizon)
        
        # Invert the scaling for the forecast and the test data
        forecasted_inverted = data_preprocessor.scaler.inverse_transform(forecasted)
        test_data_inverted  = data_preprocessor.scaler.inverse_transform(test_data)
        
        # Calculate RMSE between forecast and actual test data
        rmse = np.sqrt(np.mean((forecasted_inverted[:, 0] - test_data_inverted[:, 0]) ** 2))
        rmse_list.append(rmse)
        print(f"Fold RMSE: {rmse:.2f}")
        
    return rmse_list
