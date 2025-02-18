import numpy as np
import tensorflow as tf

class ForecastEngine:
    def __init__(self, trained_model, look_back, batch_size, neurons):
        self.trained_model = trained_model
        self.look_back = look_back
        self.batch_size = batch_size
        self.neurons = neurons

    def forecast(self, start_input, steps):
        # Define a new forecasting model
        forecast_model = Sequential()
        forecast_model.add(InputLayer(batch_input_shape=(self.batch_size, self.look_back, 1)))
        forecast_model.add(LSTM(self.neurons, activation='tanh', stateful=True, return_sequences=True))
        forecast_model.add(LSTM(self.neurons, activation='tanh', return_sequences=False))
        forecast_model.add(Dense(1))
        forecast_model.compile(loss='mean_squared_error', optimizer='adam')

        # Set weights from the trained model
        forecast_model.set_weights(self.trained_model.get_weights())

        new_predictions = []
        current_batch = start_input[-self.batch_size:]  # Get the last batch for prediction

        for i in range(steps):
            pred = forecast_model.predict(current_batch, batch_size=self.batch_size)

            for b in range(self.batch_size):
                new_predictions.append(pred[b, -1, 0])

            # Update each sequence in the batch
            new_batch = np.zeros_like(current_batch)
            for b in range(self.batch_size):
                rolled = np.roll(current_batch[b], -1)  # Shift the sequence
                rolled[-1] = pred[b, -1, 0]  # Add the new prediction
                new_batch[b] = rolled

            current_batch = new_batch

        # Convert predictions to a numpy array
        predictions_array = np.array(new_predictions).reshape(-1, 1)

        # Calculate TAF (Trend Adjusted Forecast)
        taf = self.calculate_taf(predictions_array)

        # Calculate MA (Moving Average)
        ma = self.calculate_ma(predictions_array)

        return predictions_array, taf, ma
        
