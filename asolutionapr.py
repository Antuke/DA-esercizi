import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import math

def generate_data(n=1000, n_features=1, prior=(0.5), mu_plus = 1, mu_minus = 0, var_plus = 1, var_minus = 4):
    """Generate synthetic classification dataset."""
    X = np.zeros((n, n_features))
    Y = np.zeros(n)

    for i in range(n):
        y = np.random.choice([-1, 1])
        x = np.random.normal(mu_plus, var_plus, n_features) if y == 1 else np.random.normal(mu_minus, var_minus, n_features)
        
        X[i] = x
        Y[i] = y
        

    return (X, Y)


def predict_map(x):
    sb = -15/32  * x**2 + x + (math.log(4) - 1/2)
    return 1 if sb >= 0 else -1

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

def predict_logistic(weights, X, threshold=0.5,dummy_x = False):
    """Predict class using logistic regression weights."""
    X = np.array(X).reshape(1, -1) if np.isscalar(X) else X
    if dummy_x == True:
        if np.isscalar(X):
            X = np.array([X, X**2])
        else:
            X = np.hstack((X, (X[:, 0]**2).reshape(-1, 1)))

    X_with_bias = np.insert(X, 0, 1, axis=1)
    prob = 1 / (1 + np.exp(-np.dot(X_with_bias, weights)))
    return np.where(prob >= threshold, 1, -1)


def estimate_performance(X_train, Y_train, betas, num_trials=1000,mu_plus=1,mu_minus=0,var_plus=1,var_minus=16,dummy_x=False):
    """Estimate performance across different classifiers."""
    pe_map = np.zeros(num_trials)
    pe_logistic = np.zeros(num_trials)
    pe_knn = np.zeros(num_trials)
    xtot = np.zeros(num_trials)
    ytot = np.zeros(num_trials)

    for i in range(num_trials):
        label = np.random.choice([1, -1])
        x = np.random.normal(mu_plus, var_plus) if label == 1 else np.random.normal(mu_minus, var_minus)

        xtot[i] = x
        ytot[i] = label

        pred_map = predict_map(x)
        pred_knn = KNN(X_train, Y_train, x,k=5)
        pred_logistic = predict_logistic(betas, x,dummy_x=dummy_x)

        pe_map[i] = pred_map != label
        pe_knn[i] = pred_knn != label
        pe_logistic[i] = pred_logistic[0] != label


    plot_classifier_comparison_better(xtot,ytot,betas)
    return {
        'MAP': np.mean(pe_map),
        'Logistic': np.mean(pe_logistic),
        'KNN': np.mean(pe_knn)
    }


def plot_binary_classification_data(X, y, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    
    # Make sure X is 1D
    X = X.ravel() if X.ndim > 1 else X
    
    # Plot each class using their actual labels as y-coordinates
    plt.scatter(X[y == 1], y[y == 1], c='blue', label='Class 1', alpha=0.6)
    plt.scatter(X[y == -1], y[y == -1], c='red', label='Class -1', alpha=0.6)
    
    plt.xlabel('Feature Value')
    plt.ylabel('Class Label')
    plt.title('Data Points by Class')
    plt.legend()
    plt.yticks([-1, 1])  # Only show -1 and 1 on y-axis
    plt.grid(True)  # Add grid for better visibility
    plt.show()

def plot_classifier_comparison_better(X, y, betas, figsize=(15, 5)):
    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Make X 1D if needed
    X = X.ravel() if X.ndim > 1 else X
    x_range = np.linspace(min(X) - 0.5, max(X) + 0.5, 1000)
    
    # Generalized decision function
    def decision_function(x, betas):
        return sum(beta * (x**i) for i, beta in enumerate(betas))
    
    # Logistic Regression Plot (Linear or Polynomial)
    ax1.scatter(X[y == 1], [1] * sum(y == 1), c='blue', label='Class 1', alpha=0.6)
    ax1.scatter(X[y == -1], [0] * sum(y == -1), c='red', label='Class -1', alpha=0.6)
    
    # Compute probabilities for the decision function
    probs = 1 / (1 + np.exp(-decision_function(x_range, betas)))
    ax1.plot(x_range, probs, 'black', label='Decision Function', linewidth=2)
    
    # Color the entire background based on decision regions
    ax1.fill_between(x_range, -0.1, 1.1, where=(probs < 0.5), color='red', alpha=0.2, label='Class -1 Region')
    ax1.fill_between(x_range, -0.1, 1.1, where=(probs >= 0.5), color='blue', alpha=0.2, label='Class 1 Region')
    
    ax1.set_xlabel('Feature Value')
    ax1.set_ylabel('Predicted Probability')
    ax1.set_title('Logistic Regression\n(Generalized Classifier)')
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim(-0.1, 1.1)
    
    # Quadratic Classifier Plot (Nonlinear)
    ax2.scatter(X[y == 1], [1] * sum(y == 1), c='blue', label='Class 1', alpha=0.6)
    ax2.scatter(X[y == -1], [0] * sum(y == -1), c='red', label='Class -1', alpha=0.6)
    
    # Quadratic classifier probabilities
    probs_quad = 1 / (1 + np.exp(-(-0.47 * x_range**2 + x_range + 0.886)))
    ax2.plot(x_range, probs_quad, 'black', label='Decision Function', linewidth=2)
    
    # Color the entire background based on decision regions
    ax2.fill_between(x_range, -0.1, 1.1, where=(probs_quad < 0.5), color='red', alpha=0.2, label='Class -1 Region')
    ax2.fill_between(x_range, -0.1, 1.1, where=(probs_quad >= 0.5), color='blue', alpha=0.2, label='Class 1 Region')
    
    ax2.set_xlabel('Feature Value')
    ax2.set_ylabel('Predicted Probability')
    ax2.set_title('Quadratic Classifier\n(Nonlinear Classifier)')
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.show()

def plot_classifier_comparison(X, y, betas, figsize=(15, 5)):
    
    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Make X 1D if needed
    X = X.ravel() if X.ndim > 1 else X
    x_range = np.linspace(min(X) - 0.5, max(X) + 0.5, 1000)
    
    # Logistic Regression Plot
    ax1.scatter(X[y == 1], [1]*sum(y == 1), c='blue', label='Class 1', alpha=0.6)
    ax1.scatter(X[y == -1], [0]*sum(y == -1), c='red', label='Class -1', alpha=0.6)
    
    # Linear classifier probabilities
    # QUI METTERE BETA TROVATI CON LOGISTIC REGRESSION
    probs_linear = 1 / (1 + np.exp(-(betas[0] + betas[1] * x_range )))
    ax1.plot(x_range, probs_linear, 'black', label='Decision Function', linewidth=2)
    
    ax1.set_xlabel('Feature Value')
    ax1.set_ylabel('Predicted Probability')
    ax1.set_title('Logistic Regression\n(Linear Classifier)')
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim(-0.1, 1.1)
    
    # Quadratic Classifier Plot
    ax2.scatter(X[y == 1], [1]*sum(y == 1), c='blue', label='Class 1', alpha=0.6)
    ax2.scatter(X[y == -1], [0]*sum(y == -1), c='red', label='Class -1', alpha=0.6)
    
    # Quadratic classifier probabilities
    probs_quad = 1 / (1 + np.exp(-(-0.47 * x_range**2 + x_range + 0.886)))
    ax2.plot(x_range, probs_quad, 'black', label='Decision Function', linewidth=2)
    
    ax2.set_xlabel('Feature Value')
    ax2.set_ylabel('Predicted Probability')
    ax2.set_title('Quadratic Classifier\n(Nonlinear Classifier)')
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.show()

def main():
    n_features = 1

    X, Y = generate_data(20000, n_features)

    X_dummy = np.hstack((X, (X[:, 0]**2).reshape(-1, 1)))
    
    betas = logistic_regression_sgd(X, Y)

    print(f"\nBetas = {betas}")


    performance = estimate_performance(X, Y, betas,dummy_x=False)
    print("\nPerformance Estimation:")
    for method, pe in performance.items():
        print(f"PE {method}: {pe}")
    
    plot_classifier_comparison_better(X,Y,betas)
if __name__ == "__main__":
    main()