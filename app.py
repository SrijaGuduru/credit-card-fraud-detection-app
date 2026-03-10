from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import tensorflow as tf
import torch
import torch.nn as nn

app = Flask(__name__)

# -----------------------
# Load Scaler
# -----------------------
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# -----------------------
# Load LSTM Model
# -----------------------
lstm_model = tf.keras.models.load_model("models/lstm_model.keras")

# -----------------------
# Load Autoencoder
# -----------------------
autoencoder = tf.keras.models.load_model("models/autoencoder.keras")

# -----------------------
# Liquid Neural Network
# -----------------------
class LiquidLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LiquidLayer, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(input_size, hidden_size)
        self.U = nn.Linear(hidden_size, hidden_size)
        self.tau = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        h = torch.zeros(x.size(0), self.hidden_size)
        h = torch.tanh(self.W(x) + self.U(h))
        h = h / self.tau
        return h


class LiquidNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LiquidNN, self).__init__()
        self.liquid = LiquidLayer(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h = self.liquid(x)
        return torch.sigmoid(self.output(h))


liquid_model = LiquidNN(30, 32)
liquid_model.load_state_dict(torch.load("models/liquid_model.pt"))
liquid_model.eval()

# -----------------------
# Routes
# -----------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    model_type = request.json.get("model", "lstm")

    data_np = np.array(data).reshape(1, -1)
    data_scaled = scaler.transform(data_np)

    if model_type == "lstm":
        data_lstm = data_scaled.reshape(1, 1, data_scaled.shape[1])
        prediction = lstm_model.predict(data_lstm)
        risk_score = float(prediction[0][0])

    elif model_type == "autoencoder":
        reconstruction = autoencoder.predict(data_scaled)
        mse = np.mean(np.power(data_scaled - reconstruction, 2))
        risk_score = float(mse)

    elif model_type == "liquid":
        tensor_data = torch.tensor(data_scaled, dtype=torch.float32)
        prediction = liquid_model(tensor_data)
        risk_score = float(prediction.detach().numpy()[0][0])

    else:
        return jsonify({"error": "Invalid model type"})

    result = "Fraud" if risk_score > 0.5 else "Not Fraud"

    return jsonify({
        "model_used": model_type,
        "risk_score": risk_score,
        "prediction": result
    })


if __name__ == "__main__":
    app.run(debug=True)