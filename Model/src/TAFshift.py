import numpy as np

class TAFShift:
    def __init__(self, alpha=0.5, beta=0.5):
        self.alpha = alpha  # Smoothing constant for error TAF
        self.beta = beta    # Smoothing constant for trend in TAF

    def calculate_taf(self, data, predictions):
        """full_taf, predicted_taf"""
        data = np.asarray(data).flatten()
        predictions = np.asarray(predictions).flatten()

        n = len(data) + len(predictions)

        # Create a single array with all datapoints
        combined_data = np.concatenate([data, predictions])
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
            st[t] = taf_values[t] + self.alpha*(combined_data[t] - taf_values[t])
            tt[t] = tt[t-1] + self.beta*(taf_values[t] - taf_values[t-1] - tt[t-1])

        
        # return taf_values.reshape(-1, 1), taf_values[-len(predictions_array):].reshape(-1, 1)
        return taf_values, taf_values[len(data):]
        print(taf_values[len(data):])
        # return taf_values.reshape(-1, 1), taf_values[len(data):].reshape(-1, 1)

    def apply_taf(self, historical_data, forecasted, weight=1.0, normalize=False):
        _, predicted_taf = self.calculate_taf(historical_data, forecasted)
        # We need to reshape due to applying flatten in calculate_tafm must be in (n, 1) shape for plotting
        predicted_taf = predicted_taf.reshape(-1, 1)
        # predicted_taf = predicted_taf

        # using Robust scaling
        if normalize:
            median_taf = np.median(predicted_taf)
            q1, q3 = np.percentile(predicted_taf, [22, 95])
            iqr = q3 - q1
            if iqr < 1e-6:  
                iqr = np.std(predicted_taf) if np.std(predicted_taf) > 0 else 1.0
            predicted_taf = (predicted_taf - median_taf) / iqr 
            # predicted_taf *= weight 

        # adjusted_forecast = forecasted * (1 + weight * predicted_taf)
        adjusted_forecast = forecasted + weight * predicted_taf
        return adjusted_forecast