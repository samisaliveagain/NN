import torch
from src.data_preprocessing import load_data
from src.model_dropout import MCDropoutNN
from src.train import train_model
from src.visualize_uncertainty import plot_uncertainty

X_train, X_test, y_train, y_test, scaler_y = load_data("CompositeStrengthBNN/data/Concrete_Data.csv")

# Convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

model = MCDropoutNN()
train_model(model, X_train, y_train)
plot_uncertainty(model, X_test, y_test, scaler_y)
