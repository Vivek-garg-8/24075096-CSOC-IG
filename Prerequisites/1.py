import csv
import random
import math
import matplotlib.pyplot as plt

def load_data(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        cat_idx = header.index("ocean_proximity")
        target_idx = len(header) - 1

        numeric_header = [
            col for i, col in enumerate(header)
            if i != cat_idx
        ]
        data = []
        for row in reader:
            numeric_row = [x for i, x in enumerate(row) if i != cat_idx]
            try:
                float_row = list(map(float, numeric_row))
                data.append(float_row)
            except ValueError as e:
                print(f"Skipping row {row} — {e}")
                continue

    return data, numeric_header

def normalize_data(data):
    cols = list(zip(*data))
    means = [sum(col) / len(col) for col in cols]
    stds = [math.sqrt(sum((x - mean) ** 2 for x in col) / len(col)) for col, mean in zip(cols, means)]
    norm_data = [[(x - mean) / std if std != 0 else 0 for x, mean, std in zip(row, means, stds)] for row in data]
    return norm_data, means, stds

def train_val_split(data, val_ratio=0.2):
    random.shuffle(data)
    split = int(len(data) * (1 - val_ratio))
    return data[:split], data[split:]

def predict(x, weights, bias):
    return sum(w * xi for w, xi in zip(weights, x)) + bias

def compute_cost(data, weights, bias):
    total_cost = 0
    for row in data:
        x, y = row[:-1], row[-1]
        pred = predict(x, weights, bias)
        total_cost += (pred - y) ** 2
    return total_cost / (2 * len(data))

def gradient_descent(train_data, lr=0.01, epochs=1000):
    n_features = len(train_data[0]) - 1
    weights = [0.0] * n_features
    bias = 0.0
    cost_history = []

    for epoch in range(epochs):
        dw = [0.0] * n_features
        db = 0.0

        for row in train_data:
            x, y = row[:-1], row[-1]
            pred = predict(x, weights, bias)
            error = pred - y
            for i in range(n_features):
                dw[i] += error * x[i]
            db += error

        m = len(train_data)
        weights = [w - lr * (dw[i] / m) for i, w in enumerate(weights)]
        bias -= lr * (db / m)

        cost = compute_cost(train_data, weights, bias)
        cost_history.append(cost)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Cost: {cost:.4f}")

    return weights, bias, cost_history

def evaluate(data, weights, bias):
    errors = []
    y_true = []
    y_pred = []

    for row in data:
        x, y = row[:-1], row[-1]
        pred = predict(x, weights, bias)
        y_true.append(y)
        y_pred.append(pred)
        errors.append(pred - y)

    mae = sum(abs(e) for e in errors) / len(errors)
    rmse = math.sqrt(sum(e**2 for e in errors) / len(errors))
    mean_y = sum(y_true) / len(y_true)
    r2 = 1 - (sum((yt - yp)**2 for yt, yp in zip(y_true, y_pred)) / sum((yt - mean_y)**2 for yt in y_true))

    return mae, rmse, r2

if __name__ == "__main__":
    filename = "housing.csv"
    raw_data, header = load_data(filename)
    norm_data, means, stds = normalize_data(raw_data)
    train_data, val_data = train_val_split(norm_data)

    weights, bias, cost_history = gradient_descent(train_data, lr=0.01, epochs=1000)
    print("Final weights:", weights)
    print("Final bias:", bias)

    mae, rmse, r2 = evaluate(val_data, weights, bias)
    print(f"\nEvaluation on validation set:\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR2 Score: {r2:.4f}")


    plt.plot(cost_history)
    plt.xlabel("Epoch")
    plt.ylabel("Cost (MSE/2)")
    plt.title("Gradient Descent Convergence (Pure‑Python)")
    plt.show()

