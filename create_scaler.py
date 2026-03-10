import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Drop target column
X = df.drop("Class", axis=1)

# Create scaler
scaler = StandardScaler()
scaler.fit(X)

# Save scaler
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Scaler recreated successfully!")