# main.py
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. Generate data
X = np.linspace(-5,5,200).reshape(-1,1)
y = np.sin(X) + 0.3 * np.random.randn(*X.shape)

# 2. Define model
class MLPDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1,64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64,64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64,1)
        )
    def forward(self,x): return self.net(x)

model = MLPDropout()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 3. Train
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

for epoch in range(1000):
    optimizer.zero_grad()
    loss = criterion(model(X_tensor), y_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0: print(f"Epoch {epoch}: Loss={loss.item():.4f}")

# 4. Predict with uncertainty
model.train()
preds = [model(X_tensor).detach().numpy() for _ in range(100)]
preds = np.array(preds)
mean_pred = preds.mean(axis=0)
std_pred = preds.std(axis=0)

# 5. Plot
plt.plot(X, y, 'k.', label='Data')
plt.plot(X, mean_pred, 'b', label='Mean prediction')
plt.fill_between(X.flatten(), (mean_pred-2*std_pred).flatten(), (mean_pred+2*std_pred).flatten(), color='blue', alpha=0.2, label='Uncertainty')
plt.legend(); plt.show()
