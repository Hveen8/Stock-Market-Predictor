import numpy as np

class TAFShift:
    def __init__(self, alpha=0.5, beta=0.5):
        self.alpha = alpha  # Smoothing constant for error TAF
        self.beta = beta    # Smoothing constant for trend in TAF

    def calculate_taf(self, data, predictions):
        """full_taf, predicted_taf"""
        n = len(data)
        
        # Create a single array with all datapoints
        combined_data = np.concatenate([data, predictions.flatten()])
        # combined_data = np.vstack((historical_data, predictions_array))
        total_length = len(combined_data)

        st = np.zeros(total_length) # Smooth error
        tt = np.zeros(total_length) # Trend factor
        taf_values = np.zeros(total_length)

        # Using the first timestep (from data) as initial smoothed forecast
        st[0] = combined_data[0]
        tt[0] = 0

        taf_values = np.zeros(n)
        
        # Calculate TAF values for the complete dataset:
        # https://courses.worldcampus.psu.edu/welcome/mangt515/lesson02_13.html
        for t in range(1, n):
            taf_values[t] = st[t-1] + tt[t-1]
            st[t] = taf_values[t] + alpha*(combined_data[t] - taf_values[t])
            tt[t] = tt[t-1] + beta*(taf_values[t] - taf_values[t-1] - tt[t-1])

        
        # return taf_values.reshape(-1, 1), taf_values[-len(predictions_array):].reshape(-1, 1)
        return taf_values, taf_values[len(data):]