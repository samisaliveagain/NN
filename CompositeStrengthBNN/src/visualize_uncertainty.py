import numpy as np
import matplotlib.pyplot as plt

def plot_uncertainty(model, X_test, y_test, scaler_y, num_samples=100):
    model.train()  # enable dropout
    preds = [model(X_test).detach().numpy() for _ in range(num_samples)]
    preds = np.array(preds)

    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)

    y_true = scaler_y.inverse_transform(y_test)
    y_mean = scaler_y.inverse_transform(mean_pred)

    plt.figure(figsize=(8,5))
    plt.plot(y_true, label="True Strength", color='black')
    plt.plot(y_mean, label="Predicted Strength", color='blue')
    plt.fill_between(
        range(len(y_mean.flatten())),
        (y_mean - 2 * std_pred).flatten(),
        (y_mean + 2 * std_pred).flatten(),
        color='blue', alpha=0.2, label="Uncertainty (±2σ)"
    )
    plt.legend()
    plt.title("Concrete Strength Prediction with Uncertainty")
    plt.xlabel("Sample Index")
    plt.ylabel("Compressive Strength (MPa)")
    plt.show()
