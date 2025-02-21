import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class Visualizer:
    def __init__(self, scaler: BufferedMinMaxScaler, trained_model: LSTMModel, forecast_engine: ForecastEngine):
        """Instance needs to be of || BufferedMinMaxScaler | LSTMModel | ForecastEngine || type"""
        if not isinstance(scaler, BufferedMinMaxScaler):
            raise TypeError("Expected an instance of BufferedMinMaxScaler.")
        if not isinstance(trained_model, LSTMModel):
            raise TypeError("Expected an instance of LSTMModel.")
        if not isinstance(forecast_engine, ForecastEngine):
            raise TypeError("Expected an instance of ForecastEngine.")
        self.scalar = scalar
        self.trained_model = trained_model
        self.forecast_engine = forecast_engine

    def invert_predictions(self, predictions):
        return self.scaler.inverse_transform(predictions)

    def calculate_rmse(self, true_values, predicted_values):
        return np.sqrt(mean_squared_error(true_values, predicted_values))

    def create_plot_array(self, historical, forecasts):
        # Combines historical and forecast data for plotting.
        full_length = len(historical) + len(forecasts)
        plot_array = np.full((full_length, 1), np.nan)
        plot_array[:len(historical)] = historical
        plot_array[len(historical):] = forecasts
        return plot_array

     def plot_results(self, curr_dataset, curr_system, curr_dir):
        # Extract parameters from the trained model
        look_back = self.trained_model.look_back
        batch_size = self.trained_model.batch_size
        neurons = self.trained_model.neurons
        epochs = self.trained_model.epochs
        headroom = self.trained_model.headroom
        train_predictions = self.trained_model.trainPredict

        # Extract parameters from the forecasted data
        layers = self.forecast_engine.layers
        future_predictions = self.forecast_engine.futurePredictions

        # Invert predictions
        train_predictions_inverted = self.invert_predictions(train_predictions)
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

        _dir = curr_dir
        save_dir = f"/mnt/slurm_nfs/ece498_w25_20/Stock-Market-Predictor/Model/{_dir}/"
        
        plt.savefig(f"{save_dir}{curr_system}_predictions Lr_{layers} H_{headroom} L_{look_back} B_{batch_size} N_{neurons} E_{epochs}.png")
        
        plt.close()
        
        print(f"Saved plot for column: {curr_system}")