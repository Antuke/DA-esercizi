import numpy as np
import matplotlib.pyplot as plt
from solutionmarch import logistic_regression_sgd, predict_logistic

class DataGenerator:
    """Synthetic classification dataset generator."""
    def __init__(self, n_samples=1000, n_features=1, m_good=3, m_bad=0.5):
        self.n_samples = n_samples
        self.n_features = n_features
        self.m_good = m_good
        self.m_bad = m_bad

    def generate(self):
        """Generate balanced dataset with two classes."""
        X = np.zeros((self.n_samples, self.n_features))
        Y = np.zeros(self.n_samples)
        
        # First half of the dataset
        for i in range(self.n_samples // 2):
            y = np.random.choice([-1, 1])
            x = np.random.normal(self.m_good if y == 1 else 0, 1, self.n_features)
            X[i], Y[i] = x, y
        
        # Second half of the dataset
        for i in range(self.n_samples // 2, self.n_samples):
            y = np.random.choice([-1, 1])
            x = np.random.normal(self.m_bad if y == 1 else 0, 1, self.n_features)
            X[i], Y[i] = x, y
        
        return X, Y

class Classifier:
    """MAP and Logistic Regression Classifiers."""
    @staticmethod
    def map_classifier(x, m):
        """Maximum A Posteriori classifier."""
        return 1 if 2 * m * x - m**2 > 0 else -1
    
    @staticmethod
    def posterior_plus1(x, m):
        """Compute posterior probability of class +1."""
        numerator = np.exp(-0.5 * (x - m)**2)
        denominator = numerator + np.exp(-0.5 * x**2)
        return numerator / denominator



def plot_beta(betas):
    x_values, _ = betas.shape
    print(f"xvalues = {x_values}, beta shape {betas.shape}")
    plt.figure(figsize=(10, 6))
    plt.plot(betas[:,0] , label='b0')
    plt.plot(betas[:,1], label='b1')
    plt.xlabel('iterations')
    plt.ylabel('value')
    plt.title('Beta evalotuion in SGD, lr=0.01')
    plt.legend()
    plt.grid()
    plt.show()


def run_monte_carlo_experiment(mc_iterations=10000, m_good=3, m_bad=0.5):
    """Run Monte Carlo experiment to evaluate classifiers."""
    # Generate training data
    data_generator = DataGenerator(n_samples=10000, m_good=m_good, m_bad=m_bad)
    X, Y = data_generator.generate()
    
    # Train logistic regression
    betas = logistic_regression_sgd(X, Y, learning_rate=0.1, epochs=1, weights_array=True)
    
    plot_beta(betas)



    # Initialize performance error arrays
    pe_good = np.zeros(mc_iterations)
    pe_bad = np.zeros(mc_iterations)
    pe_logistic_good = np.zeros(mc_iterations)
    pe_logistic_bad = np.zeros(mc_iterations)
    
    # Monte Carlo simulation
    for i in range(mc_iterations):
        label = np.random.choice([-1, 1])
        
        # Generate test samples
        if label == 1:
            xgood = np.random.normal(m_good, 1)
            xbad = np.random.normal(m_bad, 1)
        else:
            xgood = np.random.normal(0, 1)
            xbad = np.random.normal(0, 1)
        
        # MAP Classifier predictions
        pred_good = Classifier.map_classifier(xgood, m_good)
        pred_bad = Classifier.map_classifier(xbad, m_bad)
        
        # Logistic Regression predictions
        # Empirically selected beta indices based on previous observations
        pred_good_l = predict_logistic(betas[400], xgood)
        pred_bad_l = predict_logistic(betas[900], xbad)
        
        # Record prediction errors
        pe_good[i] = (pred_good != label)
        pe_bad[i] = (pred_bad != label)
        pe_logistic_good[i] = (pred_good_l[0] != label) # [0] è per evitare warning di numpy
        pe_logistic_bad[i] = (pred_bad_l[0] != label)
    
    # Print results
    print("\nMAP Classifier Performance Errors:")
    print(f"PE with m={m_bad}: {np.mean(pe_bad)}")
    print(f"PE with m={m_good}: {np.mean(pe_good)}")
    
    print("\nLogistic Regression Performance Errors:")
    print(f"PE with m={m_bad}: {np.mean(pe_logistic_bad)}")
    print(f"PE with m={m_good}: {np.mean(pe_logistic_good)}")

# Optional visualization (commented out in original code)
def plot_posterior_probabilities(m_good=3, m_bad=0.1):
    """Plot posterior probabilities for different means."""
    x_values = np.linspace(-10, 10, 500)
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, Classifier.posterior_plus1(x_values, m_good), label=f'm = {m_good} (good)')
    plt.plot(x_values, Classifier.posterior_plus1(x_values, m_bad), label=f'm = {m_bad} (bad)')
    plt.xlabel('x')
    plt.ylabel('p(+1|x)')
    plt.title('Posterior Probability p(+1|x)')
    plt.legend()
    plt.grid()
    plt.show()

# Run the experiment
if __name__ == "__main__":
    run_monte_carlo_experiment()
    plot_posterior_probabilities()