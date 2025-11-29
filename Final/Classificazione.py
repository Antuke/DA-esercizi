import numpy as np
import matplotlib.pyplot as plt
from math import sqrt,pi


def possterior_plus1(x, sigma):
    """Compute posterior probability of class +1."""
    numerator = np.exp( - (x - 1)**2 / (2* (sigma)**2))
    denominator = numerator + np.exp( - (x + 1)**2 / (2* (sigma)**2))
    return numerator / denominator

def posterior_plus1(x,sigma):
    """Compute posterior probability of class +1."""
    return 1 / ( 1 + np.exp(-2*x/(sigma**2)) ) 

def map_classifier(x):
    return 1 if x>0 else -1

def plot_posterior_probabilities(sigma_bad=3, sigma_good=0.5):
    """Plot posterior probabilities for different standard deviations."""
    x_values = np.linspace(-10, 10, 500)
    plt.figure(figsize=(10, 6))

    post_good = posterior_plus1(x_values, sigma_good)
    post_bad = posterior_plus1(x_values, sigma_bad)

    plt.plot(x_values, posterior_plus1(x_values, sigma_good), label=f'sigma good = {sigma_good} ')
    plt.plot(x_values, posterior_plus1(x_values, sigma_bad), label=f'sigma bad = {sigma_bad} ')

    plt.axhline(y=0.5, color='red', linestyle=':', linewidth=2, label='0.5 threshold')

    plt.fill_between(x_values, 0, 1, where=(post_good > 0.5) | (post_bad > 0.5),
                    color='lightgreen', alpha=0.3, label='Decision: +1')
    plt.fill_between(x_values, 0, 1, where=(post_good <= 0.5) & (post_bad <= 0.5),
                    color='mistyrose', alpha=0.3, label='Decision: -1')
    
    plt.xlabel('x')
    plt.ylabel('p(+1|x)')
    plt.title('Posterior Probability p(+1|x)')
    plt.legend()
    plt.grid()
    plt.show()

def likelihood_plus(x,sigma):
    return (1/sqrt(2*pi*(sigma**2)) * np.exp(- (x - 1)**2 / (2*(sigma)**2)))

def likelihood_minus(x,sigma):
    return (1/sqrt(2*pi*(sigma**2)) * np.exp(- (x + 1)**2 / (2*(sigma)**2)))

def plot_data_likelihood(sigma):
    """Plot likelihoods probabilities."""
    x_values = np.linspace(-10, 10, 500)
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, likelihood_plus(x_values, sigma), label=f'mu = 1')
    plt.plot(x_values, likelihood_minus(x_values, sigma), label=f'mu = -1')
    plt.xlabel('x')
    plt.ylabel('probabilities')
    plt.title(f'Likelihood for standard deviation {sigma}')
    plt.legend()
    plt.grid()
    plt.show()



def montecarlo(MC,sigma_good, sigma_bad, mu_plus = 1, mu_minus = -1):


    pe_bad = np.zeros(MC)
    pe_good = np.zeros(MC)
    
    for i in range(MC):

        y = np.random.choice([-1,1],p=[0.5,0.5])

        x_good = None
        x_bad = None

        if y == 1:
            x_good = np.random.normal(mu_plus,sigma_good)
            x_bad = np.random.normal(mu_plus,sigma_bad)
        else:
            x_good = np.random.normal(mu_minus,sigma_good)
            x_bad = np.random.normal(mu_minus,sigma_bad)


        pred_good = 1 if posterior_plus1(x_good,sigma_good) > 0.5 else -1
        pred_bad = 1 if posterior_plus1(x_bad, sigma_bad) > 0.5 else -1

        if pred_good != y:
            pe_good[i] = 1
        if pred_bad != y:
            pe_bad[i] = 1
    
    
    return np.mean(pe_good), np.mean(pe_bad)


def generate_data(n=1000, n_features=1, prior=(0.5), mu_plus = 1, mu_minus = -1, sigma_plus = 1, sigma_minus = 1):
    """Generate synthetic classification dataset."""
    X = np.zeros((n, n_features))
    Y = np.zeros(n)

    for i in range(n):
        y = np.random.choice([-1, 1],p=[prior,1-prior])
        x = np.random.normal(mu_plus, sigma_plus, n_features) if y == 1 else np.random.normal(mu_minus, sigma_minus, n_features)
        
        X[i] = x
        Y[i] = y
        

    return (X, Y)

def logistic_regression_sgd(X, y, learning_rate=0.03, weights_array=False, true_beta=None):
    """Stochastic Gradient Descent for Logistic Regression."""
    N, n_features = X.shape
    weights = np.zeros(n_features + 1)

    err = None
    if true_beta is not None:
        err = np.zeros(N)

    if weights_array:
        beta_array = np.zeros((N, n_features+1))

    X_with_bias = np.insert(X, 0, 1, axis=1)
    for i in range(N):
        linear_model = np.dot(X_with_bias[i], weights)
        p = 1 / (1 + np.exp(-linear_model))
        gradient =  X_with_bias[i] * (p - (y[i] == 1))
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
    X_with_bias = np.insert(X, 0, 1)
    return 1 if np.dot(X_with_bias, weights) > 0 else -1
    prob = 1 / (1 + np.exp(-np.dot(X_with_bias, weights)))


def test_error_logistic(beta,N=1000):
    X,Y = generate_data(N,1,prior=0.5,mu_plus=1,mu_minus=-1,sigma_plus=0.8,sigma_minus=0.8)
    pe_logistic = np.zeros(N)
    pe_map = np.zeros(N)
    for i in range(N):
        y_pred = predict_logistic(beta,X[i])
        y_map = 1 if posterior_plus1(X[i],0.8) > 0.5 else -1
        if y_pred != Y[i]:
            pe_logistic[i] = 1
        if y_map != Y[i]:
            pe_map[i] = 1

    return np.mean(pe_logistic),np.mean(pe_map)

def plot_error_beta(err):
    plt.plot(err)
    plt.xlabel('Iterations')
    plt.ylabel('Distance between true betas and calculated beta')
    plt.grid()
    plt.show()



def plot_beta(betas):
    x_values, _ = betas.shape
    plt.figure(figsize=(10, 6))
    plt.plot(betas[:,0] , label='b0')
    plt.plot(betas[:,1], label='b1')
    plt.xlabel('iterations')
    plt.ylabel('value')
    plt.title('Beta evalotuion in SGD, lr=0.001')
    plt.legend()
    plt.grid()
    plt.show()

def main():
    sigma_good = 0.8
    sigma_bad = 3

    test_n = 100000 # number of sample for montecarlo for MAP and for test-set of logistic
    train_n = 100000 # number of sample to train logistic regression classifier

    plot_data_likelihood(sigma_good)
    plot_data_likelihood(sigma_bad)
    plot_posterior_probabilities(sigma_good=sigma_good,sigma_bad=sigma_bad)
    pe_good, pe_bad = montecarlo(test_n,sigma_good=sigma_good,sigma_bad=sigma_bad)
    print(f"PE standard deviation {sigma_bad}  = {pe_bad}\nPE standard deviation {sigma_good}  = {pe_good}")

    X,Y = generate_data(n=train_n,sigma_plus=sigma_good,sigma_minus=sigma_good)


    beta, err = logistic_regression_sgd(X,Y,weights_array=True,true_beta=np.array([0, 2/((sigma_good)**2)]))
    plot_error_beta(err)
    plot_beta(beta)

    print(f"\nBeta = {beta[train_n-1]}")
    true_betas = [0, 2 / (sigma_good)**2]
    print(f"'True' Betas = {true_betas}")
    print(f"Distance between logistic beta and true beta = {err[train_n-1]}")
    logistic_test_error,map_err = test_error_logistic(beta[train_n-1],test_n)
    print(f"PE logistic standard deviation {sigma_good} : {logistic_test_error}")
    print(f"PE map standard deviation {sigma_good} : {map_err}")
if __name__ == "__main__":
    main()