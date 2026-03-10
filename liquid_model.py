import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class", axis=1).values
y = df["Class"].values

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


# --------------------------
# Liquid Neural Network Layer
# --------------------------
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


# --------------------------
# Full Model
# --------------------------
class LiquidNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LiquidNN, self).__init__()
        self.liquid = LiquidLayer(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h = self.liquid(x)
        out = torch.sigmoid(self.output(h))
        return out


# Initialize model
model = LiquidNN(input_size=X_train.shape[1], hidden_size=32)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "models/liquid_model.pt")
print("Liquid Neural Network trained and saved!")