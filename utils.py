import numpy as np
from scipy.spatial.distance import cdist
#------------- SGD ----------------#

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




#------------- Synthetic dataset ----------------#

def generate_data(n=1000, n_features=1, prior=(0.5), mu_plus = 1, mu_minus = -1, var_plus = 1, var_minus = 1):
    """Generate synthetic classification dataset."""
    X = np.zeros((n, n_features))
    Y = np.zeros(n)
    
    for i in range(n):
        y = np.random.choice([1, -1], p=[prior, 1 - prior])
        x = np.random.normal(mu_plus, var_plus, n_features) if y == 1 else np.random.normal(mu_minus, var_minus, n_features)
        
        X[i] = x
        Y[i] = y
        

    return (X, Y)




# ----------------- Non parametric ---------------- #

def knn_r(X_train, Y_train, x_new, k=5):
    """K-Nearest Neighbors classification. Y_train has the regression function value"""
    x_new = np.atleast_2d(x_new)
    distances = np.sqrt(np.sum((X_train - x_new) ** 2, axis=1))
    k_indices = np.argsort(distances)[:k]
    return np.sign(np.mean(Y_train[k_indices]))

def KNN(X_train,Y_train,x_new,k=1):
    """K-Nearest Neighbors classification. Y_train has the labels"""
    x_new = np.atleast_2d(x_new)
    distances = cdist(x_new, X_train).flatten()
    k_indices = np.argsort(distances)[:k]
    labels = Y_train[k_indices]
    vote = np.sum(labels)
    return 1 if vote > 0 else -1

def naive_r(X_train,Y_train,x_new,h=1):
    """Naive kernel classification. Y_train has the regression function value"""
    distances = np.sqrt(np.sum((X_train - x_new) ** 2, axis=1))
    neighbour = Y_train[distances < h]
    if len(neighbour) == 0:
        print("\n[WARNING!] No neighbour where found, try to increse h!")
        return -1 
    return np.sign(np.mean(neighbour))


def naive(X_train,Y_train,x_new,h=1):
    """Naive kernel classification. Y_train has the labels"""
    distances = np.sqrt(np.sum((X_train - x_new) ** 2, axis=1))
    neighbour = Y_train[distances < h]
    if len(neighbour) == 0:
        print("\n[WARNING!] No neighbour where found, try to increse h!")
        return -1 
    vote = np.sum(neighbour)
    return 1 if vote > 0 else -1