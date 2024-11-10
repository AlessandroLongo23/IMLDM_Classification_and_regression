import numpy as np

class BaselineClassification:
    def __init__(self, y):
        self.y = y.flatten().tolist() if isinstance(y, np.ndarray) else y
        self.most_frequent_class = None
    
    def train_(self):
        self.most_frequent_class = max(set(self.y), key=self.y.count)
        return self.most_frequent_class
    
    def eval_(self, y_test):
        # Ensure y_test is also a 1D list of hashable values
        y_test = y_test.flatten().tolist() if isinstance(y_test, np.ndarray) else y_test
        
        # Predict the most frequent class for all test samples
        predictions = [self.most_frequent_class] * len(y_test)
        
        # Compute the number of misclassified samples
        num_misclassified = sum(1 for y_true, y_pred in zip(y_test, predictions) if y_true != y_pred)
        
        # Compute error rate
        E = num_misclassified / len(y_test)
        return E