from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

def train_autoencoder(data):
    input_dim = data.shape[1]

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(8, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='linear')(encoded)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(data, data, epochs=20, batch_size=8, verbose=0)

    return autoencoder
