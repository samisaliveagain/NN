# import torch
# from src.data_preprocessing import load_data
# from src.model_dropout import MCDropoutNN
# from src.train import train_model
# from src.visualize_uncertainty import plot_uncertainty

# X_train, X_test, y_train, y_test, scaler_y = load_data("CompositeStrengthBNN/data/Concrete_Data.csv")

# # Convert to torch tensors
# X_train = torch.tensor(X_train, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.float32)
# X_test = torch.tensor(X_test, dtype=torch.float32)
# y_test = torch.tensor(y_test, dtype=torch.float32)

# model = MCDropoutNN()
# train_model(model, X_train, y_train)
# plot_uncertainty(model, X_test, y_test, scaler_y)







import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.data_preprocessing import load_data
from src.model_dropout import MCDropoutNN
from src.train import train_model
from src.visualize_uncertainty import plot_uncertainty

# ================================================
# 1. Load and prepare data
# ================================================
X_train, X_test, y_train, y_test, scaler_y = load_data("CompositeStrengthBNN/data/Concrete_Data.csv")

# Convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# ================================================
# 2. Define and train the model
# ================================================
model = MCDropoutNN()
train_model(model, X_train, y_train)

# ================================================
# 3. Visualize model uncertainty
# ================================================
plot_uncertainty(model, X_test, y_test, scaler_y)

# ================================================
# 4. Evaluate model performance (new section)
# ================================================
model.eval()
with torch.no_grad():
    y_pred = model(X_test).detach().numpy()

# Inverse transform both to original MPa scale
y_pred_original = scaler_y.inverse_transform(y_pred)
y_test_original = scaler_y.inverse_transform(y_test)

# Compute metrics
mse = mean_squared_error(y_test_original, y_pred_original)
mae = mean_absolute_error(y_test_original, y_pred_original)
r2 = r2_score(y_test_original, y_pred_original)

print("\n================ Model Performance ================")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")
print("===================================================")

# ================================================
# 5. Scatter plots (new section)
# ================================================

# (a) True vs Predicted — model fit visualization
plt.figure(figsize=(6,6))
plt.scatter(y_test_original, y_pred_original, color='blue', alpha=0.6, label='Predicted Points')
plt.plot(
    [y_test_original.min(), y_test_original.max()],
    [y_test_original.min(), y_test_original.max()],
    'r--', label='Perfect Prediction (y=x)'
)
plt.xlabel("Actual Compressive Strength (MPa)")
plt.ylabel("Predicted Compressive Strength (MPa)")
plt.title("True vs Predicted Concrete Strength")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# (b) Original data distribution for show
plt.figure(figsize=(6,6))
plt.scatter(range(len(y_test_original)), y_test_original, color='green', alpha=0.5, label='Actual Strength')
plt.scatter(range(len(y_pred_original)), y_pred_original, color='orange', alpha=0.5, label='Predicted Strength')
plt.title("Concrete Strength: Actual vs Predicted")
plt.xlabel("Sample Index")
plt.ylabel("Compressive Strength (MPa)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()








