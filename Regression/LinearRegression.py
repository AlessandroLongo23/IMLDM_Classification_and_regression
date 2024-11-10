import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

class LinearRegression:
    def __init__(self, X, y, include_bias=True):
        if not include_bias:
            self.X = X.astype(np.float64)
        else:
            self.X = np.concatenate((np.ones((X.shape[0], 1)), X.astype(np.float64)), axis=1)
            
        self.y = y.astype(np.float64)
        self.weights = np.zeros((self.X.shape[1], 1))
        self.include_bias = include_bias
        
    def add_bias(self, X):
        return np.hstack((np.ones((X.shape[0], 1)), X))
        
    def remove_bias(self, X):
        return X[:, 1:]
        
    def solve_analytical(self, lambda_=0.0):
        n_features = self.X.shape[1]
        identity = np.eye(n_features)
        if self.include_bias:
            identity[0, 0] = 0  # Do not regularize the bias term
        
        XtX = np.dot(self.X.T, self.X)
        XtX_reg = XtX + lambda_ * identity
        Xty = np.dot(self.X.T, self.y)
    
        self.weights = np.linalg.solve(XtX_reg, Xty)   
        
    def predict(self):
        return np.dot(self.X, self.weights)
    
    def score(self):
        y_pred = self.predict()
        
        ss_total = np.sum((self.y - np.mean(self.y)) ** 2)
        ss_residual = np.sum((self.y - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        
        mse = np.mean((self.y - y_pred) ** 2)
        
        return {'R2': r2, 'MSE': mse}
    
    def plot_regularization_effects(self, lambda_range, n_samples):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        lambdas = np.logspace(lambda_range[0], lambda_range[1], n_samples) 

        # Plot 1: Regularization Path
        paths = []
        for lambda_ in lambdas:
            self.solve_analytical(lambda_)
            paths.append(self.weights.flatten())

        paths = np.array(paths)
        for i in range(paths.shape[1]):
            ax1.semilogx(lambdas, paths[:, i], '-', label='Bias Term' if i == 0 else f'Feature {i}')
        
        ax1.set_xlabel('λ (Regularization Parameter)')
        ax1.set_ylabel('Weight Value')
        ax1.set_title('Weight Values vs Regularization λ')
        ax1.grid(True, which="both", ls="-", alpha=0.2)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 2: Model Performance
        r2_scores = []
        mse_scores = []
        for lambda_ in lambdas:
            self.solve_analytical(lambda_)
            scores = self.score()
            r2_scores.append(scores['R2'])
            mse_scores.append(scores['MSE'])
        
        # Plot R² score
        ax2.semilogx(lambdas, r2_scores, 'b-', label='R² Score')
        ax2.set_xlabel('λ (Regularization Parameter)')
        ax2.set_ylabel('R² Score', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        
        # Plot MSE on secondary y-axis
        ax3 = ax2.twinx()
        ax3.semilogx(lambdas, mse_scores, 'r-', label='MSE')
        ax3.set_ylabel('Mean Squared Error', color='r')
        ax3.tick_params(axis='y', labelcolor='r')
        
        # Add legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        
        ax2.set_title('Model Performance vs Regularization')
        ax2.grid(True, which="both", ls="-", alpha=0.2)
        
        # Adjust layout
        plt.tight_layout()
        plt.show()
        
    def one_level_cross_validation(self, lambda_, K=10):
        kf = KFold(n_splits=K, shuffle=True, random_state=42)
        
        mse_scores = []
        
        # Remove bias term from overall X and add it back to each fold
        X_no_bias = self.remove_bias(self.X) if self.include_bias else self.X
        
        for train_index, test_index in kf.split(X_no_bias):
            X_train, X_test = X_no_bias[train_index], X_no_bias[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            if self.include_bias:
                # Add bias term manually back to each fold of X_train and X_test
                X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
                X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
            
            fold_model = LinearRegression(X_train, y_train, include_bias=False)
            fold_model.solve_analytical(lambda_)
            
            y_pred = np.dot(X_test, fold_model.weights)
            mse = np.mean((y_test - y_pred) ** 2)
            mse_scores.append(mse)
            
        generalization_error = np.mean(mse_scores)
        
        return generalization_error
    
    def plot_one_level_cv_generalization_error(self, lambda_range=(-3, 6), num_lambdas=30, K=10):
        generalization_errors = [{
            'lambda': lambda_,
            'err': self.one_level_cross_validation(lambda_, K),   
        } for lambda_ in np.logspace(lambda_range[0], lambda_range[1], num_lambdas)]
        
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        
        opt_lambda = np.inf
        lowest_mse = np.inf
        
        for i in generalization_errors:
            if i['err'] < lowest_mse:
                lowest_mse = i['err']
                opt_lambda = i['lambda']
        print(f'Optimal lambda: {opt_lambda}')
        print(f'Lowest MSE: {lowest_mse}')
        
        ax.semilogx([r['lambda'] for r in generalization_errors], [r['err'] for r in generalization_errors], '-')
        ax.set_xlabel('lambda')
        ax.set_ylabel('Generalization error')
        ax.set_title('Generalization error vs. lambda')
        
        plt.show()