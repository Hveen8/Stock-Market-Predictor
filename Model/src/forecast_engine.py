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

    def forecast(self, start_input, steps, target_col_idx=0):
        # Use _build_model from LSTMModel to create a new model for forecasting
        forecast_model = self._build_model(self.batch_size, forecast_horizon=steps)  # Rebuild model for forecasting

        # Set weights from the trained model
        forecast_model.set_weights(self.trained_model.model.get_weights())

        new_predictions = []
        current_batch = start_input[-self.batch_size:]  # Get the last batch for prediction
        # current_batch = [current_batch.copy() for _ in range(self.batch_size)]
        # last_look_back_seq = start_input[-1]

        # print('Future batch steps: ', math.ceil( steps/self.batch_size ))
        # for i in range(math.ceil( steps/self.batch_size )):
        # # ===================================================================
        for i in range(steps):
        #     # # (1, look_back, num_features)
        #     # current_input = np.expand_dims(last_look_back_seq, axis=0)
        #     # # Can do this (batch_size=1) since have Dense(1)
        #     # pred = forecast_model.predict(current_input, batch_size=1)
        #     # next_target_value = pred[0, 0]
            
        #     # new_predictions.append(next_target_value)


        #     # new_step = np.copy(last_look_back_seq[-1])
        #     # # Replace the target column with the prediction. Other features stay the same
        #     # new_step[target_col_idx] = next_target_value


        #     # last_look_back_seq = np.vstack((last_look_back_seq[1:], new_step))


            pred = forecast_model.predict(current_batch, batch_size=self.batch_size)

            new_predictions.append(pred[-1, -1, target_col_idx])

        #     # Get all the next infered values/timesteps, put into 3D shape
            new_step = pred[:, -1, :].reshape(self.batch_size, 1, -1)
            # new_step = pred[:, -1:]

        #     # for b in range(self.batch_size):
        #     #     new_predictions.append(pred[b, -1, 0])
        #     # print('Predictions shape: ', pred[i, -1, :].shape)
		#     # print('New Inference (Prediction): ', pred[-1, -1, 0])

        #     # # Update each sequence in the batch
        #     # new_batch = np.zeros_like(current_batch)
        #     # for b in range(self.batch_size):
        #     #     # Shift the sequence
        #     #     rolled = np.roll(current_batch[b], -1)
        #     #     # Add the new prediction at the end
        #     #     rolled[-1] = pred[b, -1, 0]
        #     #     new_batch[b] = rolled
        #     # current_batch = new_batch

        #     # Update of batch for next prediction step, dropping the oldest value (in look back) and
        #     # adding the new infered values (newest in look back)
            current_batch = np.concatenate([current_batch[:, 1:, :], new_step], axis=1)
        # ===================================================================

        # # Multi-Step Forecasting (Takes advantage of the TimeDistributed Dense layer)
        # multi_step_forecast = forecast_model.predict(current_batch, batch_size=self.batch_size)
        # print("multi_step_forecast shape:", multi_step_forecast.shape)
        # new_predictions = multi_step_forecast[0]
        # # new_predictions = multi_step_forecast[-1, :, target_col_idx]
        # # new_predictions = np.mean(multi_step_forecast[:, :, target_col_idx], axis=0)
        # print("new_predictions shape:", new_predictions.shape)

        # Convert predictions to a numpy array (predictions_array)
        self.futurePredictions = np.array(new_predictions).reshape(-1, 1)
        # self.futurePredictions = new_predictions

        # ***** May need to manage resetting the states better, however may not be necessary 

        return self.futurePredictions