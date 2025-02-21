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

    def create_dataset(self, dataset, look_back=1):
        dataX, dataY = [], []

        for i in range(len(dataset)-look_back-1):
            # Note the 0 here indicated to put into a 1D array [1, 2, 3]
            # Instead of 2D [[1], [2], [3]]
            a = dataset[i:(i+look_back), 0]
            # Appened into shape(X, 3)
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])

        return np.array(dataX), np.array(dataY)

    def trim_XY(self, dataX, dataY, batch_size):
        # Removing any odd data depending on batch size
        trim_size = len(dataX) - (len(dataX) % batch_size)
        return dataX[:trim_size], dataY[:trim_size]

    def create_input_layer(self, batch_size, look_back):
        # store the generated input layer into the class
        self.input_layer = InputLayer(batch_input_shape=(batch_size, look_back, 1))
        return self.input_layer