import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class Visualizer:
    def __init__(self):
    # def __init__(self, scaler: BufferedMinMaxScaler):
    #     """Instance needs to be of BufferedMinMaxScaler type"""
    #     if not isinstance(scaler, BufferedMinMaxScaler):
    #         raise TypeError("Expected an instance of BufferedMinMaxScaler.")
    #     # Create an instance of Visualizer with the scaler
    #     # EXAMPLE: visualizer = Visualizer(scaler)
    #     self.scaler = scaler

    def invert_predictions(self, scaler: BufferedMinMaxScaler, predictions):
        """Instance needs to be of BufferedMinMaxScaler type"""
        if not isinstance(scaler, BufferedMinMaxScaler):
            raise TypeError("Expected an instance of BufferedMinMaxScaler.")
        return scaler.inverse_transform(predictions)

    def calculate_rmse(self, true_values, predicted_values):
        return np.sqrt(mean_squared_error(true_values, predicted_values))

    def create_plot_array(self, historical, forecasts):
        # Combines historical and forecast data for plotting.
        full_length = len(historical) + len(forecasts)
        plot_array = np.full((full_length, 1), np.nan)
        plot_array[:len(historical)] = historical
        plot_array[len(historical):] = forecasts
        return plot_array

     def plot_results(self, trained_model, forecast_engine, curr_dataset, future_predictions, curr_system):
        """Plots historical data along with training and future predictions."""
        
        # Extract parameters from the trained model and forecast engine
        look_back = trained_model.look_back
        batch_size = trained_model.batch_size
        neurons = trained_model.neurons
        epochs = trained_model.epochs  # or any other relevant parameter

        # Invert predictions
        train_predictions_inverted = self.invert_predictions(trained_model.trainPredict)
        historical_data_inverted = self.invert_predictions(curr_dataset)
        future_predictions_inverted = self.invert_predictions(future_predictions)

        # Calculate RMSE
        train_rmse = self.calculate_rmse(historical_data_inverted[look_back:], train_predictions_inverted)
        
        print('Train Score: %.2f RMSE' % (train_rmse))

        # Shift train predictions for plotting
        train_predict_plot = np.empty_like(historical_data_inverted)
        train_predict_plot[:, :] = np.nan
        train_predict_plot[look_back:len(train_predictions_inverted) + look_back] = train_predictions_inverted

        # Create full time array for x-axis
        full_time = np.arange(len(historical_data_inverted) + len(future_predictions_inverted))

        # Create plot array with NaNs
        plot_array = self.create_plot_array(historical_data_inverted, future_predictions_inverted)

        # Plotting
        plt.figure(figsize=(12, 6))
        
        plt.plot(full_time[:len(historical_data)], historical_data_inverted, color='blue', linewidth=1.5, label='Given Data')
        plt.plot(full_time[look_back:len(train_predict_plot) + look_back], train_predict_plot[look_back:], color='green', alpha=0.75, label='Training Data (Prediction)')
        
        # Plot future predictions
        plt.plot(full_time[len(historical_data):], future_predictions_inverted, color='red', linestyle='--', linewidth=1.5, label='Future Predictions')

        # Add labels and title
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(f'LSTM Predictions vs. Given Data for {curr_system}')
        
        plt.legend()

        # Save the figure
        save_dir = f'/mnt/slurm_nfs/ece498_w25_20/test_slurm5_L2_{trained_model.headroom}_results_batched/'
        
        plt.savefig(f"{save_dir}{curr_system}_predictions (H_{trained_model.headroom}) L_{look_back} B={batch_size} N={neurons} E={epochs}.png")
        
        plt.close()
        
        print(f"Saved plot for column: {curr_system}")