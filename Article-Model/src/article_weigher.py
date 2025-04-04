import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

class ArticleWeightPredictor:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = self._build_model()
        
    def _build_model(self):
        """
        Constructs a neural network mapping embeddings to four outputs:
           - Relevance score (0 to 1)
           - Direction weights (stock up, stock down, stock unchanged) that sum to 1
        """
        input_layer = layers.Input(shape=(self.input_dim,))
        # Shared hidden layers
        x = layers.Dense(128, activation='relu')(input_layer)
        x = layers.Dense(64, activation='relu')(x)
        
        relevance = layers.Dense(1, activation='sigmoid', name='relevance')(x)
        
        direction_logits = layers.Dense(3, name='direction_logits')(x)
        # Using softmax so the 3 ebeddings from the layer will add up to 1
        direction = layers.Activation('softmax', name='direction')(direction_logits)
        
        output = layers.Concatenate(name='output')([relevance, direction])
        
        model = models.Model(inputs=input_layer, outputs=output)
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.summary()
        return model
    
    def train(self, x, y, epochs=50, batch_size=32, validation_split=0.2):
        """
        Parameters:
          - X: Array of embeddings (shape: [num_samples, input_dim])
          - y: Array of target outputs (shape: [num_samples, 4])
               where the first element is the relevance score, and the next three are directional weights.
        """
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=validation_split, random_state=42)
        history = self.model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=3)
            ]
        )
        return history
    
    def predict(self, embedding):
        """Output is 4 dimensional"""
        return self.model.predict(np.array([embedding]))
    
    def save_model(self, filepath):
        self.model.save(filepath)
    
    @classmethod
    def load_model(cls, filepath, input_dim):
        predictor = cls(input_dim)
        predictor.model = tf.keras.models.load_model(filepath)
        return predictor
