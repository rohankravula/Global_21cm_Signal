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

# Split data (60% train, 20% validation, 20% test)
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Initialize model and error tracking
print("Training models with different n_estimators...")
train_errors = []
val_errors = []
test_errors = []
n_estimators_range = range(1, 201)  # From 1 to 200

model = RandomForestRegressor(max_depth=15, random_state=42, n_jobs=-1)

for i in n_estimators_range:
    model.set_params(n_estimators=i)
    model.fit(X_train, y_train)
    
    # Predict on training, validation, and test sets
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Calculate errors
    train_errors.append(mean_squared_error(y_train, y_train_pred))
    val_errors.append(mean_squared_error(y_val, y_val_pred))
    test_errors.append(mean_squared_error(y_test, y_test_pred))
    
    # Print progress
    if i % 20 == 0 or i == 1:
        print(f"n_estimators={i}: Train MSE={train_errors[-1]:.2f}, Val MSE={val_errors[-1]:.2f}")

# Find optimal n_estimators (minimum validation error)
optimal_n = n_estimators_range[np.argmin(val_errors)]
print(f"\nOptimal n_estimators: {optimal_n}")
print(f"Minimum Validation MSE: {min(val_errors):.4f}")

# Create output directory
output_dir = os.path.join(os.path.expanduser("~"), "Downloads", "model_analysis")
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Plot the error curve
plt.figure(figsize=(12, 6))
plt.plot(n_estimators_range, train_errors, 'b-', label='Training Error')
plt.plot(n_estimators_range, val_errors, 'r-', label='Validation Error')
plt.plot(n_estimators_range, test_errors, 'g-', label='Test Error')
plt.axvline(x=optimal_n, color='k', linestyle='--', label=f'Optimal n_estimators ({optimal_n})')
plt.title('Error vs Number of Trees', fontsize=14)
plt.xlabel('Number of Trees (n_estimators)', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

error_curve_path = os.path.join(output_dir, f"error_curve_{timestamp}.png")
plt.savefig(error_curve_path, dpi=300, bbox_inches='tight')
plt.close()

# Generate ROC-style curve for regression (using error distribution)
plt.figure(figsize=(10, 8))
errors = y_test - y_test_pred
sorted_idx = np.argsort(y_test)
plt.plot(np.linspace(0, 1, len(y_test)), np.sort(errors), 'b-', label='Error Distribution')
plt.axhline(y=0, color='k', linestyle='--', label='Perfect Prediction')
plt.title('Error Distribution Curve (Testing Set)', fontsize=14)
plt.xlabel('Percentile', fontsize=12)
plt.ylabel('Prediction Error (Actual - Predicted)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

roc_path = os.path.join(output_dir, f"error_distribution_{timestamp}.png")
plt.savefig(roc_path, dpi=300, bbox_inches='tight')
plt.close()

# Train final model with optimal n_estimators on combined train+val data
print("\nTraining final model with optimal n_estimators...")
final_model = RandomForestRegressor(n_estimators=optimal_n, max_depth=15, random_state=42, n_jobs=-1)
final_model.fit(np.vstack((X_train, X_val)), np.concatenate((y_train, y_val)))

# Evaluate final model
y_test_pred_final = final_model.predict(X_test)
final_mse = mean_squared_error(y_test, y_test_pred_final)
final_mae = mean_absolute_error(y_test, y_test_pred_final)
print(f"\nFinal Test MSE: {final_mse:.4f}")
print(f"Final Test MAE: {final_mae:.4f}")

# Generate and save sampled actual vs predicted scatter plot for final model
print("Generating sampled actual vs predicted plot...")
sample_size = 500
sample_indices = np.random.choice(len(y_test), size=sample_size, replace=False)
y_test_sample = y_test[sample_indices]
y_pred_sample = y_test_pred_final[sample_indices]

plt.figure(figsize=(10, 10))
plt.scatter(y_test_sample, y_pred_sample, alpha=0.6, color='teal')
plt.plot([min(y_test_sample), max(y_test_sample)], 
         [min(y_test_sample), max(y_test_sample)], 'k--', lw=2)
plt.title(f'Actual vs Predicted Values (Sample of {sample_size})\nFinal Model (n_estimators={optimal_n})', fontsize=14)
plt.xlabel('Actual Y', fontsize=12)
plt.ylabel('Predicted Y', fontsize=12)
plt.grid(True, alpha=0.3)

final_plot_path = os.path.join(output_dir, f"final_prediction_{timestamp}.png")
plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\nError curve saved to: {error_curve_path}")
print(f"Error distribution curve saved to: {roc_path}")
print(f"Final prediction plot saved to: {final_plot_path}")

# Print sample predictions from final model
print("\nSample Actual vs Predicted (Final Model):")
sample_indices = np.random.choice(len(y_test), size=5, replace=False)
for i in sample_indices:
    print(f"Input: A={X_test[i,0]:.2f}, b={X_test[i,1]:.2f}, x={X_test[i,2]:.2f}")
    print(f"  Actual: {y_test[i]:.4f} | Predicted: {y_test_pred_final[i]:.4f}\n")