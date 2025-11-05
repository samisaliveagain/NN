import torch
from torch import optim, nn
import matplotlib.pyplot as plt
import numpy as np
from dataset import generate_data
from model import MLPDropout

# Data
X, y = generate_data()
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Model
model = MLPDropout()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Evaluation with uncertainty
model.train()  # Keep dropout ON
preds = []
for _ in range(100):
    preds.append(model(X_tensor).detach().numpy())
preds = np.array(preds)

mean_pred = preds.mean(axis=0)
std_pred = preds.std(axis=0)

plt.figure(figsize=(8,4))
plt.plot(X, y, 'k.', label='Data')
plt.plot(X, mean_pred, 'b', label='Mean prediction')
plt.fill_between(X.flatten(), (mean_pred-2*std_pred).flatten(), (mean_pred+2*std_pred).flatten(), color='blue', alpha=0.2, label='Uncertainty')
plt.legend()
plt.savefig("results/uncertainty_plot.png")
plt.show()
