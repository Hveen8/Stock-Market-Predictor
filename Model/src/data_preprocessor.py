import numpy as np
from sklearn.preprocessing import MinMaxScaler

class BufferedMinMaxScaler(MinMaxScaler):
    """New Subclass, inheriting from MinMaxScalar, gives custom/buffered scale"""
    def __init__(self, headroom=0.2):
        super().__init__()
        self.headroom = headroom

    def fit(self, X, y=None):
        X = np.asarray(X)
        # 1. Store original data min/max
        self.orig_data_min_ = X.min(axis=0)
        self.orig_data_max_ = X.max(axis=0)
        #self.data_min_ = X.min(axis=0)
        #self.data_max_ = X.max(axis=0)

        # 2. Calculate buffer-adjusted max
        data_range = self.orig_data_max_ - self.orig_data_min_
        self.data_max_ = self.orig_data_max_ + data_range * self.headroom
        self.data_min_ = self.orig_data_min_  # Keep original min (for now, potentially will need to change)

        # 3. Calculate parent class parameters (data_range_ is for parent class, incorporating the headroom)
        self.data_range_ = self.data_max_ - self.data_min_
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / self.data_range_
        self.min_ = self.feature_range[0] - self.data_min_ * self.scale_
        return self 

class DataPreprocessor:
    def __init__(self, headroom=0.2):
        # Its using the extra headroom by defult
        self.scaler = BufferedMinMaxScaler(headroom=headroom)

    def fit_transform(self, data):
        return self.scaler.fit_transform(data)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def create_dataset(self, dataset, look_back=1, target_feature=0, forecast_horizon=1):
        dataX, dataY = [], []

        for i in range(len(dataset)-look_back-1):
            # Note the : here indicated to put into a 2D array [[1,2,3], [1,2,3], [1,2,3]]
            input = dataset[i:(i+look_back), :] # IF there was only 1 feature, then need 0 to put into 1D
            # Appened into shape(X, 3)
            dataX.append(input)
            output = dataset[i + look_back, :]
            # output = dataset[i+look_back:i+look_back+forecast_horizon, :]
            dataY.append(output)

        return np.array(dataX), np.array(dataY)

    def trim_XY(self, dataX, dataY, batch_size):
        # Removing any odd data depending on batch size
        trim_size = len(dataX) - (len(dataX) % batch_size)
        return dataX[:trim_size], dataY[:trim_size]

    def invert_1d_prediction(self, pred_1d, feature_cols_num, target_feature=0):
        # If we are looking at ALL the included features
        if pred_1d.shape[1] == feature_cols_num:
            inverted = self.scaler.inverse_transform(pred_1d)
        else:
            # pred_1d shape: (samples,1) -> what the INFERENCE output of the LSTM is
            # (samples, num_features) -> (samples,1) keeping target column
            padded = np.zeros((pred_1d.shape[0], feature_cols_num), dtype=np.float32)
            # [0, 0, 1, 0]
            # [0, 0, 1, 0] <- is essentially this
            # [0, 0, 1, 0]
            if pred_1d.shape[1] == 1:
                padded[:, target_feature] = pred_1d[:, 0]
            else:
                padded[:, target_feature] = pred_1d[:, target_feature]
            inverted = self.scaler.inverse_transform(padded)
        # Return just the "target" column (out of all the feature columns)
        return inverted[:, target_feature].reshape(-1,1)

def filter_multi_features(dataset, stock_rows, feature_cols):
    df_symbol = dataset[dataset['symbol'] == stock_rows].copy()
    df_symbol[feature_cols] = df_symbol[feature_cols].fillna(0)
    full_dataset = df_symbol[feature_cols].values.astype('float32')
    return full_dataset