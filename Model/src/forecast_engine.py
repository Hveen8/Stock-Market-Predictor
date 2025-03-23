import math
import numpy as np
import tensorflow as tf
from .lstm_model import LSTMModel

class ForecastEngine(LSTMModel):
    def __init__(self, trained_model, layers=None, isReturnSeq=True, features=None, look_back=None, batch_size=None, neurons=None, epochs=None, activation=None, dropout=None):
        params = {'layers': layers,
                'isReturnSeq' : isReturnSeq,
                'features' : features,
                'look_back': look_back,
                'batch_size': batch_size,
                'neurons': neurons,
                'epochs': epochs,
                'activation': activation,
                'dropout': dropout}

        # Use parameters from trained_model or default values IF specified
        for param_name in params:
            if params[param_name] is None:
                params[param_name] = getattr(trained_model, param_name)

        # Initialize LSTMModel with inherited parameters
        super().__init__(layers=params['layers'],
                         isReturnSeq=params['isReturnSeq'],
                         features=params['features'],
                         look_back=params['look_back'],
                         batch_size=params['batch_size'],
                         neurons=params['neurons'],
                         epochs=params['epochs'],
                         activation=params['activation'],
                         dropout=params['dropout'])

        self.trained_model = trained_model
        # What the class will fill
        self.futurePredictions = None

    def forecast(self, start_input, steps, target_col_idx):
        # # Define a new forecasting model
        # forecast_model = Sequential()
        # forecast_model.add(InputLayer(batch_input_shape=(self.batch_size, self.look_back, 1)))
        # # really will only use 2 layers...
        # if self.layers > 1:
        #     for l in range(self.layers-1):
        #         forecast_model.add(LSTM(self.neurons, activation=self.activation, stateful=True, return_sequences=True))
        #     # In multi-layered the last is non-stateful ***
        #     forecast_model.add(LSTM(self.neurons, activation=self.activation, return_sequences=True))
        # else:
        #     forecast_model.add(LSTM(neurons, activation=self.activation, stateful=True, return_sequences=True))
        # forecast_model.add(Dense(1))
        # forecast_model.compile(loss='mean_squared_error', optimizer='adam')

        # Use _build_model from LSTMModel to create a new model for forecasting
        forecast_model = self._build_model()  # Rebuild model for forecasting

        # Set weights from the trained model
        forecast_model.set_weights(self.trained_model.model.get_weights())

        new_predictions = []
        # current_batch = start_input[-self.batch_size:]  # Get the last batch for prediction
        last_look_back_seq = start_input[-1]

        # print('Future batch steps: ', math.ceil( steps/self.batch_size ))
        # for i in range(math.ceil( steps/self.batch_size )):
        for i in range(steps):
            # (1, look_back, num_features)
            current_input = np.expand_dims(last_look_back_seq, axis=0)
            # Can do this (batch_size=1) since have Dense(1)
            pred = forecast_model.predict(current_input, batch_size=1)
            next_target_value = pred[0, 0]
            
            new_predictions.append(next_target_value)


            new_step = np.copy(last_look_back_seq[-1])
            # Replace the target column with the prediction. Other features stay the same
            new_step[target_col_idx] = next_target_value


            current_sequence = np.vstack((current_sequence[1:], new_step))

            # for b in range(self.batch_size):
            #     new_predictions.append(pred[b, -1, 0])
            # # print('Predictions shape: ', pred[i, -1, :].shape)
		    # # print('New Inference (Prediction): ', pred[-1, -1, 0])

            # # Update each sequence in the batch
            # new_batch = np.zeros_like(current_batch)
            # for b in range(self.batch_size):
            #     # Shift the sequence
            #     rolled = np.roll(current_batch[b], -1)
            #     # Add the new prediction at the end
            #     rolled[-1] = pred[b, -1, 0]
            #     new_batch[b] = rolled

            # current_batch = new_batch

        # Convert predictions to a numpy array (predictions_array)
        self.futurePredictions = np.array(new_predictions).reshape(-1, 1)

        # ***** May need to manage resetting the states better, however may not be necessary 

        return self.futurePredictions

    # def calculate_taf(self, predictions_array, historical_data, alpha, beta):
    #     """full_taf, predicted_taf"""
    #     # Combine historical and predicted data
    #     combined_data = np.vstack((historical_data, predictions_array))
    #     n = len(combined_data)

    #     # Smooth error
    #     st = np.zeros(n)
    #     # Trend factor
    #     tt = np.zeros(n)
        
    #     # Using the first timestep (from data) as initial smoothed forecast
    #     st[0] = combined_data[0]
    #     tt[0] = 0
        
    #     taf_values = np.zeros(n)
        
    #     for t in range(1, n):
    #         taf_values[t] = st[t - 1] + tt[t - 1]
    #         st[t] = taf_values[t] + alpha*(combined_data[t] - taf_values[t])
    #         tt[t] = tt[t - 1] + beta*(taf_values[t] - taf_values[t - 1] - tt[t - 1])
        
    #     # Return only TAF values corresponding to predictions
    #     return taf_values.reshape(-1, 1), taf_values[-len(predictions_array):].reshape(-1, 1)

    # def calculate_ma(self, predictions_array, ma_window):
    #     """Calculate Moving Average (MA)."""
    #     n = len(predictions_array)
        
    #     if n < self.ma_window:
    #         print("Not enough data to calculate MA")
    #         return np.full((n,), np.nan)
        
    #     ma_values = np.convolve(predictions_array.flatten(), np.ones(self.ma_window)/self.ma_window, mode='valid')
        
    #     # Pad with NaN for alignment with original predictions length
    #     ma_full = np.full(predictions_array.shape[0], np.nan)
    #     ma_full[self.ma_window-1:] = ma_values
        
    #     # output array contains MA values for positions where the entire window fits within input data
    #     # it is essentially the length: n - window_size
    #     # The end is the average of the window (so not to the end of the dataset)
    #     return ma_full.reshape(-1, 1)

    # def adjust_predictions_with_taf(self, predictions_array, historical_data):
    #     taf_values, taf_pred_values = self.calculate_taf(predictions_array, historical_data)
        
    #     # The TAF is shifting each timestep from the predictions, given a shift wieght
    #     adjusted_predictions_taf = predictions_array + taf_values
        
    #     return adjusted_predictions_taf

    # def adjust_predictions_with_ma(self, predictions_array):
    #     ma_values, ma_pred_values = self.calculate_ma(predictions_array)
        
    #     adjusted_predictions_ma = predictions_array + ma_pred_values
        
    #     return adjusted_predictions_ma