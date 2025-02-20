import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer

class LSTMModel:
    def __init__(self, look_back, batch_size, neurons, epochs, is2Layer, activation, dropout):
        self.look_back = look_back
        self.batch_size = batch_size
        self.neurons = neurons
        self.epochs = epochs
        self.is2Layer = is2Layer
        self.activation = activation # should be either the str 'tanh' or 'relu'
        self.dropout = dropout
        self.model = self._build_model()
        # What the class will fill
        self.trainPredict = None

    def _build_model(self):
        model = Sequential()
        # batch_input_shape (batch_size, num_steps, features)
        model.add(InputLayer(batch_input_shape=(self.batch_size, self.look_back, 1)))
        # the more complex the data -> more neurons needed
        if self.is2Layer:
            model.add(LSTM(self.neurons, activation=self.activation, dropout=self.dropout, stateful=True, return_sequences=True))
            model.add(LSTM(self.neurons, activation=self.activation, dropout=self.dropout, return_sequences=False))
        else:
            model.add(LSTM(neurons, activation=self.activation, stateful=True, return_sequences=False))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

        model.summary()

    def train(self, trainX, trainY):
        for layer in model.layers:
            if isinstance(layer, LSTM):
                layer.reset_states()

        for i in range(self.epochs):
            # Unlike CNNs, for RNNs (LSTM fitted) we do not shuffle
            model.fit(trainX, trainY, 
                        epochs=1, 
                        batch_size=self.batch_size, 
                        shuffle=False, 
                        verbose=2)
            
            # Resetting states after each epoch for stateful LSTM
            for layer in model.layers:
                if isinstance(layer, LSTM):
                    layer.reset_states()
            print(f"Epoch {i+1}/{epochs} --- Completed")

    def predict(self, trainX):
        # X is the training data
        self.trainPredict = self.model.predict(trainX, batch_size=self.batch_size)
        return self.trainPredict