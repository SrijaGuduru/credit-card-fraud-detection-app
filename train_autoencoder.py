import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle

df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train = X_scaled[y == 0]

input_dim = X_train.shape[1]

input_layer = layers.Input(shape=(input_dim,))
encoded = layers.Dense(16, activation="relu")(input_layer)
encoded = layers.Dense(8, activation="relu")(encoded)
decoded = layers.Dense(16, activation="relu")(encoded)
decoded = layers.Dense(input_dim, activation="linear")(decoded)

autoencoder = models.Model(input_layer, decoded)
autoencoder.compile(optimizer="adam", loss="mse")

autoencoder.fit(X_train, X_train,
                epochs=10,
                batch_size=256,
                shuffle=True)

autoencoder.save("models/autoencoder.keras")
print("Autoencoder trained!")