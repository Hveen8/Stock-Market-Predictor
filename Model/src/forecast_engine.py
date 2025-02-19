import numpy as np
import tensorflow as tf

class ForecastEngine:
    def __init__(self, trained_model, look_back, batch_size, neurons, is2Layer):
        self.trained_model = trained_model
        self.look_back = look_back
        self.batch_size = batch_size
        self.neurons = neurons
        self.is2Layer = is2Layer

    def forecast(self, start_input, steps):
        # Define a new forecasting model
        forecast_model = Sequential()
        forecast_model.add(InputLayer(batch_input_shape=(self.batch_size, self.look_back, 1)))
        if self.is2Layer:
            model.add(LSTM(self.neurons, activation=self.activation, stateful=True, return_sequences=True))
            model.add(LSTM(self.neurons, activation=self.activation, return_sequences=True))
        else:
            model.add(LSTM(neurons, activation=self.activation, stateful=True, return_sequences=True))
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
                # Shift the sequence
                rolled = np.roll(current_batch[b], -1)
                # Add the new prediction at the end
                rolled[-1] = pred[b, -1, 0]
                new_batch[b] = rolled

            current_batch = new_batch

        # Convert predictions to a numpy array
        predictions_array = np.array(new_predictions).reshape(-1, 1)

        return predictions_array

    def calculate_taf(self, predictions, historical_data):
        # Combine historical and predicted data
        combined_data = np.vstack((historical_data, predictions))
        n = len(combined_data)

        # Smoothed forecast
        st = np.zeros(n)
        # Trend estimate
        tt = np.zeros(n)
        
        st[0] = combined_data[0]  # Initial smoothed forecast is the first prediction
        tt[0] = 0  # Initial trend estimate is zero
        
        taf_values = np.zeros(n)  # To store TAF values
        
        for t in range(1, n):
            taf_values[t] = st[t - 1] + tt[t - 1]
            st[t] = taf_values[t] + self.alpha * (combined_data[t] - taf_values[t])
            tt[t] = tt[t - 1] + self.beta * (taf_values[t] - taf_values[t - 1] - tt[t - 1])
        
        return taf_values[-len(predictions):].reshape(-1, 1)  # Return only TAF values corresponding to predictions