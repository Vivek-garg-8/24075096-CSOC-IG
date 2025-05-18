import csv
import time
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

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
    return data[~np.isnan(data).any(axis=1)], [header[i] for i in num_indices]

def normalize_features(X):
    means = X.mean(axis=0)
    stds  = X.std(axis=0)
    return (X - means) / stds, means, stds

def train_val_split(X, y, val_ratio=0.2, seed=42):
    np.random.seed(seed)
    m = X.shape[0]
    idx = np.random.permutation(m)
    split = int(m * (1 - val_ratio))
    return X[idx[:split]], y[idx[:split]], X[idx[split:]], y[idx[split:]]

if __name__ == "__main__":
    data, cols = load_numeric_data("housing.csv")
    X_raw, y_raw = data[:, :-1], data[:, -1]

    X, feat_means, feat_stds = normalize_features(X_raw)
    y_mean, y_std = y_raw.mean(), y_raw.std()
    y = (y_raw - y_mean) / y_std

    X_train, y_train, X_val, y_val = train_val_split(X, y)

    model = LinearRegression()
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0

    y_pred_norm = model.predict(X_val)
    mae_norm   = mean_absolute_error(y_val, y_pred_norm)
    rmse_norm  = root_mean_squared_error(y_val, y_pred_norm)
    r2         = r2_score(y_val, y_pred_norm)

    mae_doll  = mae_norm * y_std
    rmse_doll = rmse_norm * y_std

    print(f"\n[sklearn] Fit time: {fit_time:.4f} s")
    print(f"[sklearn] Validation MAE (norm):   {mae_norm:.4f}")
    print(f"[sklearn] Validation RMSE (norm):  {rmse_norm:.4f}")
    print(f"[sklearn] Validation R²:           {r2:.4f}")
    print(f"[sklearn] Validation MAE (dollars):  {mae_doll:.2f}")
    print(f"[sklearn] Validation RMSE (dollars): {rmse_doll:.2f}")
