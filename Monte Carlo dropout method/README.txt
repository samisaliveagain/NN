1. Generating synthetic data

2. Defining a neural network with dropout

3. Training the model

4. Performing Monte Carlo sampling to estimate uncertainty

5. Visualizing predictions and uncertainty bounds








The code Creates 200 evenly spaced points between -5 and 5 (X).

Generates corresponding target values y from a sine function with added Gaussian noise (0.3 * random.randn) to simulate real-world noisy data.

reshape(-1,1) ensures each sample has a single input feature (shape [200, 1]).








The model is a 3-layer fully connected neural network (a simple MLP).

Dropout (0.1) randomly deactivates 10% of neurons during training.
This adds stochasticity, helping prevent overfitting and allowing uncertainty estimation when kept active during inference.

Layers:

Linear(1, 64) — Maps one input feature to 64 neurons.

ReLU() — Introduces non-linearity.

Dropout(0.1) — Randomly drops 10% of neurons.

Second hidden layer (64→64) repeats this structure.

Linear(64, 1) — Outputs a single predicted value.










Convert numpy arrays → torch.Tensor for compatibility with PyTorch.

Each epoch performs:

Forward pass: model(X_tensor) computes predictions.

Loss computation: criterion(...) measures prediction error.

Backward pass: loss.backward() computes gradients via backpropagation.

Optimizer step: optimizer.step() updates parameters.

Every 100 epochs, it prints training loss to track progress.







The code lacks 

1. Train/validation data split

2. Model saving/loading

3. Batch training (uses full dataset each epoch)

4. Early stopping or scheduler

5. Scalable data pipeline (e.g., DataLoader)