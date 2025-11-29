import numpy as np



def generate_data(n=1000, n_features=1, prior=(0.5), mu_plus = 1, mu_minus = 0, var_plus = 1, var_minus = 16):
    """Generate synthetic classification dataset."""
    X = np.zeros((n, n_features))
    Y = np.zeros(n)

    for i in range(n):
        y = np.random.choice([1, -1], p=[prior, 1 - prior])
        x = np.random.normal(mu_plus, var_plus, n_features) if y == 1 else np.random.normal(mu_minus, var_minus, n_features)
        
        X[i] = x
        Y[i] = y
        

    return (X, Y)


def predict_map(x):
    return 1 if (7*x - 1.75) > 0 else -1


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



def KNN(X_train,Y_train,x_new,k=1):
    """K-Nearest Neighbors classification."""
    x_new = np.atleast_2d(x_new)
    distances = np.sqrt(np.sum((X_train - x_new) ** 2, axis=1))
    k_indices = np.argsort(distances)[:k]
    labels = Y_train[k_indices]
    vote = np.sum(labels)
    return 1 if vote > 0 else -1



def predict_logistic(betas,X):
    X = np.array(X).reshape(1, -1) if np.isscalar(X) else X
    X_with_bias = np.insert(X, 0, 1, axis=1)
    sb = np.dot(X_with_bias,betas)
    return 1 if sb>0 else -1


def estimate_performance(X_train, Y_train, betas, num_trials=10000,prior=0.5,mu_plus=1,mu_minus=0,var_plus=1,var_minus=1):
    """Estimate performance across different classifiers."""
    pe_map = np.zeros(num_trials)
    pe_logistic = np.zeros(num_trials)
    pe_knn = np.zeros(num_trials)
    xtot = np.zeros(num_trials)
    ytot = np.zeros(num_trials)

    for i in range(num_trials):
        label = np.random.choice([1, -1],p=[prior,1-prior])
        x = np.random.normal(mu_plus, var_plus) if label == 1 else np.random.normal(mu_minus, var_minus)

        xtot[i] = x
        ytot[i] = label

        pred_map = predict_map(x)
        pred_knn = KNN(X_train, Y_train, x,k=2)
        pred_logistic = predict_logistic(betas, x)

        pe_map[i] = pred_map != label
        pe_knn[i] = pred_knn != label
        pe_logistic[i] = pred_logistic != label


    return {
        'MAP': np.mean(pe_map),
        'Logistic': np.mean(pe_logistic),
        'KNN': np.mean(pe_knn)
    }


def main():
    X,Y = generate_data(n=10000,n_features=1,prior=0.5,mu_plus=2,mu_minus=-1.5,var_plus=1,var_minus=1)
    betas = logistic_regression_sgd(X, Y)

    print(f"\nBetas = {betas}")


    performance = estimate_performance(X_train=X, Y_train=Y, betas=betas, num_trials=10000,prior=0.5,mu_plus=2,mu_minus=-1.5,var_plus=1,var_minus=1)
    print("\nPerformance Estimation:")
    for method, pe in performance.items():
        print(f"PE {method}: {pe}")
    
if __name__ == "__main__":
    main()