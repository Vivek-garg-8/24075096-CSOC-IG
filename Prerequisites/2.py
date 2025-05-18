import csv
import time
import numpy as np

def load_numeric_data(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        cat_idx = header.index("ocean_proximity")
        num_indices = [i for i in range(len(header)) if i != cat_idx]

    data = np.genfromtxt(
        filename, delimiter=',', skip_header=1,
        usecols=num_indices,  
        filling_values=np.nan
    )
    data = data[~np.isnan(data).any(axis=1)]
    return data, [header[i] for i in num_indices]

def normalize_features(X):
    means = X.mean(axis=0)
    stds  = X.std(axis=0)
    X_norm = (X - means) / stds
    return X_norm, means, stds

def train_val_split(X, y, val_ratio=0.2, seed=42):
    np.random.seed(seed)
    m = X.shape[0]
    idx = np.random.permutation(m)
    split = int(m * (1 - val_ratio))
    train_idx, val_idx = idx[:split], idx[split:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

def compute_cost_vec(X, y, w, b):
    m = X.shape[0]
    errors = X.dot(w) + b - y
    return (errors @ errors) / (2 * m)

def gradient_descent_vec(X, y, lr=0.01, epochs=1000):
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0
    cost_history = np.zeros(epochs)

    start_time = time.time()
    for epoch in range(epochs):
        preds = X.dot(w) + b
        error = preds - y

        dw = (X.T @ error) / m
        db = error.sum() / m

        w -= lr * dw
        b -= lr * db

        cost_history[epoch] = (error @ error) / (2 * m)

        if epoch % 100 == 0:
            print(f"[NumPy] Epoch {epoch}, Cost: {cost_history[epoch]:.4f}")

    total_time = time.time() - start_time
    return w, b, cost_history, total_time

def evaluate_metrics(X, y, w, b):
    preds = X.dot(w) + b
    errors = preds - y
    mae  = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    r2   = 1 - (errors @ errors) / ((y - y.mean()) @ (y - y.mean()))
    return mae, rmse, r2

if __name__ == "__main__":
    data, cols = load_numeric_data("housing.csv")
    X_raw, y_raw = data[:, :-1], data[:, -1]

    X_norm, means, stds = normalize_features(X_raw)

    y_mean, y_std = y_raw.mean(), y_raw.std()
    y_norm = (y_raw - y_mean) / y_std

    X_train, y_train, X_val, y_val = train_val_split(X_norm, y_norm)

    w_np, b_np, cost_hist_np, time_np = gradient_descent_vec(
        X_train, y_train, lr=0.01, epochs=1000
    )
    print(f"\n[NumPy] Training time: {time_np:.3f} seconds")

    mae_np, rmse_np, r2_np = evaluate_metrics(X_val, y_val, w_np, b_np)

    mae_dollars  = mae_np * y_std
    rmse_dollars = rmse_np * y_std

    print(f"\n[NumPy] Validation MAE (norm):        {mae_np:.4f}")
    print(f"[NumPy] Validation RMSE (norm):      {rmse_np:.4f}")
    print(f"[NumPy] Validation R²:               {r2_np:.4f}")
    print(f"[NumPy] Validation MAE (dollars):    {mae_dollars:.2f}")
    print(f"[NumPy] Validation RMSE (dollars):   {rmse_dollars:.2f}")

