import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def generate_data(n=1000, n_features=1, data_type='logistic_knn'):
    """Generate synthetic classification dataset."""
    X = np.zeros((n, n_features))
    Y = np.zeros(n)
    r = np.zeros(n) if data_type == 'logistic_knn' else None

    for i in range(n):
        y = np.random.choice([-1, 1])
        x = np.random.normal(2, 1, n_features) if y == 1 else np.random.normal(0, 1, n_features)
        
        X[i] = x
        Y[i] = y
        
        if data_type == 'logistic_knn':
            r[i] = (2*x[0] - 2) + np.random.normal(0, 1)
    
    return (X, Y, r) if data_type == 'logistic_knn' else (X, Y)

def predict_map(x):
    """Maximum A Posteriori prediction."""
    return 1 if x > 1 else -1

def logistic_regression_sgd(X, y, learning_rate=0.01, epochs=1, weights_array=False, true_beta=None):
    """Stochastic Gradient Descent for Logistic Regression."""
    N, n_features = X.shape
    weights = np.zeros(n_features + 1)

    err = None
    if true_beta is not None:
        err = np.zeros(N)

    if weights_array:
        beta_array = np.zeros((N, n_features+1))

    X_with_bias = np.insert(X, 0, 1, axis=1)
    for _ in range(epochs):
        for i in range(N):
            linear_model = np.dot(X_with_bias[i], weights) # statistica di decsione
            gradient = -y[i] * X_with_bias[i] / (1 + np.exp(y[i] * linear_model))
            weights -= learning_rate * gradient
            if weights_array:
                beta_array[i, :] = weights
            if true_beta is not None:
                err[i] =  np.linalg.norm(true_beta-weights)**2


    if true_beta is not None and weights_array:
        return beta_array, err
    if true_beta is not None:
        return weights, err
    if weights_array:
        return beta_array

    return weights

def predict_logistic(weights, X, threshold=0.5):
    """Predict class using logistic regression weights."""
    X = np.array(X).reshape(1, -1) if np.isscalar(X) else X
    X_with_bias = np.insert(X, 0, 1, axis=1)
    return 1 if np.dot(X_with_bias, weights) > 0 else -1
    prob = 1 / (1 + np.exp(-np.dot(X_with_bias, weights)))
    return np.where(prob >= threshold, 1, -1)

def knn(X_train, Y_train, x_new, k=5):
    """K-Nearest Neighbors classification."""
    x_new = np.atleast_2d(x_new)
    distances = np.sqrt(np.sum((X_train - x_new) ** 2, axis=1))
    k_indices = np.argsort(distances)[:k]
    return np.sign(np.mean(Y_train[k_indices]))

def naive(X_train,Y_train,x_new,h=1):
    distances = np.sqrt(np.sum((X_train - x_new) ** 2, axis=1))
    neighbour = Y_train[distances < h]
    if len(neighbour) == 0:
        print("\n[WARNING!] No neighbour where found, try to increse h!")
        return -1 
    return np.sign(np.mean(neighbour))

def estimate_performance(X_train, Y_train, r_train, betas, num_trials=10000):
    """Estimate performance across different classifiers."""
    pe_map = np.zeros(num_trials)
    pe_logistic = np.zeros(num_trials)
    pe_knn = np.zeros(num_trials)
    pe_naive = np.zeros(num_trials)

    for i in range(num_trials):
        label = np.random.choice([1, -1])
        x = np.random.normal(2, 1) if label == 1 else np.random.normal(0, 1)

        pred_map = predict_map(x)
        pred_knn = knn(X_train, r_train, x,k=1)
        pred_logistic = predict_logistic(betas, x)
        pred_naive = naive(X_train,r_train,x,h=1)
        
        pe_map[i] = pred_map != label
        pe_knn[i] = pred_knn != label
        pe_naive[i] = pred_naive != label
        pe_logistic[i] = pred_logistic != label

    return {
        'MAP': np.mean(pe_map),
        'Logistic': np.mean(pe_logistic),
        'KNN': np.mean(pe_knn),
        'Naive': np.mean(pe_naive)
    }

def main():
    n_features = 1
    X, Y, r = generate_data(10000, n_features, 'logistic_knn')
    betas = logistic_regression_sgd(X,Y)

    print(f"\nBetas = {betas}")
    performance = estimate_performance(X, Y, r, betas)
    print("\nPerformance Estimation:")
    for method, pe in performance.items():
        print(f"PE {method}: {pe}")

if __name__ == "__main__":
    main()