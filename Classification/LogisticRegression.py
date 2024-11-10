import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, X, y, learning_rate=0.01, max_iterations=1000, tol=1e-4, lambda_=0, include_bias=True, silent=True):
        if not include_bias:
            self.X = X.astype(np.float64)
        else:
            self.X = np.hstack((np.ones((X.shape[0], 1)), X.astype(np.float64)))
            
        self.y = y.astype(np.float64).reshape(-1, 1)
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tol = tol
        self.lambda_ = lambda_
        self.silent = silent
        
        # Initialize weights with proper shape
        self.weights = np.zeros((self.X.shape[1], 1))
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def train_(self):
        m = len(self.y)
        prev_loss = float('inf')
        
        for iteration in range(self.max_iterations):
            # Forward pass
            z = np.dot(self.X, self.weights)
            y_pred = self.sigmoid(z)
            
            # Compute loss (binary cross-entropy)
            loss = -np.mean(
                self.y * np.log(np.clip(y_pred, 1e-15, 1 - 1e-15)) +
                (1 - self.y) * np.log(np.clip(1 - y_pred, 1e-15, 1 - 1e-15))
            ) + (self.lambda_ / (2 * m)) * np.sum(self.weights[1:] ** 2)

            if abs(prev_loss - loss) < self.tol:
                if not self.silent:
                    print(f"Converged after {iteration} iterations")
                break
            prev_loss = loss
            
            if iteration % 100 == 0 and not self.silent:
                print(f"Iteration {iteration}, Loss: {loss}")
            
            gradient = (1 / m) * np.dot(self.X.T, (y_pred - self.y))
            gradient[1:] += (self.lambda_ / m) * self.weights[1:]
            
            self.weights -= self.learning_rate * gradient
            
    def eval_(self, X_test, y_test):
        y_test = y_test.astype(np.float64).reshape(-1, 1)
        y_pred = self.predict(X_test)
        
        error_rate = 1 - np.mean(y_pred == y_test)
        if not self.silent:
            print(f"Error rate: {error_rate:.4f}")
        
        return error_rate
    
    def predict_probability(self, X):
        if X is None:
            X = self.X
            
        if X.shape[1] != self.weights.shape[0]:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        return self.sigmoid(np.dot(X, self.weights))
    
    def predict(self, X, threshold=0.5):
        probas = self.predict_probability(X)
        return (probas >= threshold).astype(int)
    
    def one_level_cross_validation(self, lambda_, K=10):
        kf = KFold(n_splits=K, shuffle=True, random_state=42)
        
        error_scores = []
        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            
            fold_model = LogisticRegression(X_train, y_train, lambda_=lambda_)
            fold_model.train_()

            error_scores.append(fold_model.eval_(X_test, y_test))
            
        generalization_error = np.mean(error_scores)
        return generalization_error
    
    def plot_one_level_cv_generalization_error(self, lambda_range=(-3, 6), num_lambdas=50, K=10):
        generalization_errors = [{
            'lambda': lambda_,
            'err': self.one_level_cross_validation(lambda_, K),   
        } for lambda_ in np.logspace(lambda_range[0], lambda_range[1], num_lambdas)]
        
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        
        opt_lambda = 1000
        lowest_mse = 1000
        
        for i in generalization_errors:
            if i['err'] < lowest_mse:
                lowest_mse = i['err']
                opt_lambda = i['lambda']
        print(f'Optimal lambda: {opt_lambda}')
        print(f"Optimal lambda (log10): {np.log10(opt_lambda):.2f}")
        print(f'Lowest MSE: {lowest_mse}')
        
        ax.semilogx([r['lambda'] for r in generalization_errors], [r['err'] for r in generalization_errors], '-')
        ax.set_xlabel('lambda')
        ax.set_ylabel('Generalization error')
        ax.set_title('Generalization error vs. lambda')
        
        plt.show()