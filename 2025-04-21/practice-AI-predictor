import math
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Generate dataset
def generate_dataset(num_points=10000):
    data = []
    for _ in range(num_points):
        A = random.uniform(1, 100)
        b = random.uniform(1, 100)
        x = random.uniform(1, 100)
        Y = A * (math.sin(x) + b)
        data.append([A, b, x, Y])
    return pd.DataFrame(data, columns=["A", "b", "x", "Y"])

# Load data
print("Generating dataset...")
df = generate_dataset(10000)

# Features and target
X = df[["A", "b", "x"]].values
y = df["Y"].values

# Split 80% train, 20% test
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model
print("Training model...")
model = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Predict on test
print("Making predictions...")
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"\nTest MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")

# Print sample predictions
print("\nSample Actual vs Predicted:")
for i in range(5):
    print(f"Input: A={X_test[i,0]:.2f}, b={X_test[i,1]:.2f}, x={X_test[i,2]:.2f}")
    print(f"  Actual: {y_test[i]:.4f} | Predicted: {y_pred[i]:.4f}\n")

# Create timestamped filename for Downloads folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
downloads_path = os.path.expanduser("~/Downloads")
filename = f"actual_vs_predicted_{timestamp}.png"
filepath = os.path.join(downloads_path, filename)

# Generate and save sampled actual vs predicted scatter plot
print("Generating sampled actual vs predicted plot...")
sample_size = 500  # You can adjust this number as needed
sample_indices = np.random.choice(len(y_test), size=sample_size, replace=False)
y_test_sample = y_test[sample_indices]
y_pred_sample = y_pred[sample_indices]

plt.figure(figsize=(10, 10))
plt.scatter(y_test_sample, y_pred_sample, alpha=0.6, color='teal')
plt.plot([min(y_test_sample), max(y_test_sample)], [min(y_test_sample), max(y_test_sample)], 'k--', lw=2)
plt.title(f'Actual vs Predicted Values (Sample of {sample_size})')
plt.xlabel('Actual Y')
plt.ylabel('Predicted Y')
plt.grid(True)
plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.close()

print(f"\nSampled scatter plot saved to: {filepath}")
