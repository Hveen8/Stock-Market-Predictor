import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from .data_preprocessor import BufferedMinMaxScaler
from .lstm_model import LSTMModel
from .forecast_engine import ForecastEngine

class Visualizer:
    def __init__(self, scaler: 'BufferedMinMaxScaler', trained_model: 'LSTMModel', forecast_engine: 'ForecastEngine'):
        """Instance needs to be of || BufferedMinMaxScaler | LSTMModel | ForecastEngine || type"""
        if not isinstance(scaler, BufferedMinMaxScaler):
            raise TypeError("Expected an instance of BufferedMinMaxScaler.")
        if not isinstance(trained_model, LSTMModel):
            raise TypeError("Expected an instance of LSTMModel.")
        if not isinstance(forecast_engine, ForecastEngine):
            raise TypeError("Expected an instance of ForecastEngine.")
        self.scaler = scaler
        self.trained_model = trained_model
        self.forecast_engine = forecast_engine

    # def invert_predictions(self, predictions):
    #     return self.scaler.inverse_transform(predictions)

    # def calculate_rmse(self, true_values, predicted_values):
    #     return np.sqrt(mean_squared_error(true_values, predicted_values))

    def create_plot_array(self, full_length, loaction_num, value_arr):
        # full_length = len(historical) + len(forecasts)
        plot_array = np.zeros((len(full_length), 1))
        plot_array[loaction_num:loaction_num+len(value_arr)] = value_arr
        return plot_array

    # def plot_results(self, rmse, train_predictions_inverted, train_end, test_end, future_predictions_inverted, curr_dataset, curr_system, curr_dir):
    def plot_results(self, rmse, train_predictions_inverted, train_end, test_end, future_predictions_inverted, curr_dataset, curr_system, curr_dir, TAFvars):
        """Both Train and Forecast must be given INVERTED, following the .predict/forecast output"""
        # Extract parameters from the trained model
        look_back = self.trained_model.look_back
        batch_size = self.trained_model.batch_size
        neurons = self.trained_model.neurons
        epochs = self.trained_model.epochs
        dropout = self.trained_model.dropout
        train_predictions = self.trained_model.trainPredict

        # Extract parameters from the forecasted data
        layers = self.forecast_engine.layers
        future_predictions = self.forecast_engine.futurePredictions

        # Extract parameters from the scaler
        headroom = self.scaler.headroom

        # Invert transformations
        # train_predictions_inverted = self.invert_predictions(train_predictions)
        # historical_data_inverted = self.invert_predictions(curr_dataset) -> curr_dataset was never transformed
        # wrapping train_Y into a list -> to make it 2D, which invert_predictions requires
        # train_Y_inverted = self.invert_predictions([train_Y])
        # future_predictions_inverted = self.invert_predictions(future_predictions)

        # Calculate RMSE (Of the Training prediction, not forecast)
        # train_rmse = self.calculate_rmse(historical_data_inverted[look_back:], train_predictions_inverted)
        # train_rmse = self.calculate_rmse(train_Y_inverted[0], train_predictions_inverted[:, 0])
        # print('Train Score: %.2f RMSE' % (train_rmse))

        # Create full time array for x-axis
        full_time = np.arange(len(curr_dataset) + len(future_predictions_inverted))
        # full_time = np.arange(train_end + len(future_predictions_inverted))

        # Create plot array for plot values
        # plot_array_train = self.create_plot_array(full_time, len(curr_dataset)-len(train_predictions_inverted), train_predictions_inverted)
        plot_array_forecast = self.create_plot_array(full_time, train_end, future_predictions_inverted)

        # Plotting
        plt.figure(figsize=(12, 6))

        plt.plot(full_time[:train_end], curr_dataset[:train_end], color='blue', linewidth=1.5, label='Given Data (TRAIN)')
        plt.plot(full_time[train_end:len(curr_dataset)], curr_dataset[train_end:len(curr_dataset)], color='green', linewidth=1.5, alpha=0.95, label='Given Data (TEST)')
        # plt.plot(full_time[:len(curr_dataset)], plot_array_train[:len(curr_dataset)], color='green', linewidth=1.0, alpha=0.75, label='Training Data (Prediction)')
        plt.plot(full_time[train_end:test_end], plot_array_forecast[train_end:test_end], color='red', linestyle='--', linewidth=1.5, alpha=0.75, label='Future Predictions')

        # Add labels and title
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(f'{curr_system}, RMSE: {rmse:.2f} | TAF A:{TAFvars[0]} B:{TAFvars[1]} W:{TAFvars[2]}')
        
        plt.legend()

        # save_dir = f"/mnt/slurm_nfs/ece498_w25_20/Stock-Market-Predictor/Model/{curr_dir}/"
        save_dir = f"{curr_dir}/"
        
        plt.savefig(f"{save_dir}{curr_system}_predictions Lr_{layers} H_{headroom} N_{neurons} B_{batch_size} L_{look_back} E_{epochs} D_{dropout}.png")
        
        plt.close()
        
        print(f"Saved plot for column: {curr_system}")