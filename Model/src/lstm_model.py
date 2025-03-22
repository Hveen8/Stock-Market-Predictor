import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer

class LSTMModel:
    def __init__(self, layers, look_back, batch_size, neurons, epochs, activation, dropout, isReturnSeq=False):
        self.layers = layers
        self.isReturnSeq = isReturnSeq # should be either True or False
        self.look_back = look_back
        self.batch_size = batch_size
        self.neurons = neurons
        self.epochs = epochs
        self.activation = activation # should be either the str 'tanh' or 'relu'
        self.dropout = dropout
        self.model = self._build_model()
        # What the class will fill
        self.trainPredict = None

    def _build_model(self):
        # SHOULD ONLY HAVE ONE MODEL IN MEMORY (TF handles the models in memory in a funny (funny = i dont know))
        # So clear_session is to clear the way tf stores/handles the models
        tf.keras.backend.clear_session()
        model = Sequential()
        # batch_input_shape (batch_size, num_steps, features)
        model.add(InputLayer(batch_input_shape=(self.batch_size, self.look_back, 1)))
        # the more complex the data -> more neurons needed
        if self.layers > 1:
            for l in range(self.layers-1):
                model.add(LSTM(self.neurons, activation=self.activation, dropout=self.dropout, stateful=True, return_sequences=True))
            # In multi-layered the last is non-stateful ***
            model.add(LSTM(self.neurons, activation=self.activation, dropout=self.dropout, return_sequences=self.isReturnSeq))
        else:
            model.add(LSTM(self.neurons, activation=self.activation, dropout=self.dropout, stateful=True, return_sequences=self.isReturnSeq))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.summary()
        return model

    # def create_input_layer(self, batch_size, look_back):
    #     # store the generated input layer into the class
    #     self.input_layer = InputLayer(batch_input_shape=(batch_size, look_back, 1))
    #     return self.input_layer

    def reset_model_states(self):
        for layer in self.model.layers:
            if isinstance(layer, LSTM):
                layer.reset_states()
        print('Model states reset')

    def train(self, trainX, trainY):
        self.reset_model_states()

        for i in range(self.epochs):
            # Unlike CNNs, for RNNs (LSTM fitted) we do not shuffle
            self.model.fit(trainX, trainY, 
                        epochs=1, 
                        batch_size=self.batch_size, 
                        shuffle=False, 
                        verbose=2)
            
            # Resetting states after each epoch for stateful LSTM
            self.reset_model_states()
            print(f"Epoch {i+1}/{self.epochs} --- Completed")

    def predict(self, trainX):
        # X is the training data
        self.trainPredict = self.model.predict(trainX, batch_size=self.batch_size)
        return self.trainPredict