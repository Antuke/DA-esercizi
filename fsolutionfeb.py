import numpy as np
from scipy.spatial.distance import cdist


def generate_data(n=1000, n_features=1, prior=(0.5), mu_plus = 0.5, mu_minus = -0.5):
    """Generate synthetic classification dataset."""
    X = np.zeros((n, n_features))
    Y = np.zeros(n)

    for i in range(n):
        y = np.random.choice([-1, 1])
        x = np.random.normal(mu_plus, 1, n_features) if y == 1 else np.random.normal(mu_minus, 1, n_features)
        
        X[i] = x
        Y[i] = y
        

    return (X, Y)


def predict_map(x):
    return 1 if x > 0 else -1

def KNN(X_train,Y_train,x_new,k=1):
    """K-Nearest Neighbors classification."""
    x_new = np.atleast_2d(x_new)
    distances = cdist(x_new, X_train).flatten()
    k_indices = np.argsort(distances)[:k]
    labels = Y_train[k_indices]
    vote = np.sum(labels)
    return 1 if vote > 0 else -1


def logistic_regression_sgd(X, y, learning_rate=0.01, epochs=1, weights_array=False):
    """Stochastic Gradient Descent for Logistic Regression."""
    N, n_features = X.shape
    weights = np.zeros(n_features + 1)
    beta_array = None
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

    if weights_array:
        return beta_array
    return weights

def predict_logistic(weights, X, threshold=0.5):
    """Predict class using logistic regression weights."""
    X = np.array(X).reshape(1, -1) if np.isscalar(X) else X
    X_with_bias = np.insert(X, 0, 1, axis=1)
    prob = 1 / (1 + np.exp(-np.dot(X_with_bias, weights)))
    return np.where(prob >= threshold, 1, -1)


def estimate_performance(X_train, Y_train, betas, num_trials=10000):
    """Estimate performance across different classifiers."""
    pe_map = np.zeros(num_trials)
    pe_logistic = np.zeros(num_trials)
    pe_knn = np.zeros(num_trials)
    mu_plus = 0.5
    mu_minus = -0.5
    for i in range(num_trials):
        label = np.random.choice([1, -1])
        x = np.random.normal(mu_plus, 1) if label == 1 else np.random.normal(mu_minus, 1)

        pred_map = predict_map(x)
        pred_knn = KNN(X_train, Y_train, x,k=1)
        pred_logistic = predict_logistic(betas, x)

        pe_map[i] = pred_map != label
        pe_knn[i] = pred_knn != label
        pe_logistic[i] = pred_logistic[0] != label

    return {
        'MAP': np.mean(pe_map),
        'Logistic': np.mean(pe_logistic),
        'KNN': np.mean(pe_knn)
    }


def main():
    n_features = 1
    X, Y = generate_data(5000, n_features)
    betas = logistic_regression_sgd(X, Y)
    print(f"\nBetas = {betas}")
    performance = estimate_performance(X, Y, betas)
    print("\nPerformance Estimation:")
    for method, pe in performance.items():
        print(f"PE {method}: {pe}")

if __name__ == "__main__":
    main()