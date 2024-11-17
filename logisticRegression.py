import pandas as pd
import numpy as np

class LogisticRegression:

    def __init__(self, learning_rate=0.001, num_iter=100000):
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))    

    def fit(self , X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        
        for _ in range(self.num_iter):
            linear_model = np.dot(X , self.weights) + self.bias
            predictions = self.sigmoid(linear_model)

            dw = (1/num_samples) * np.dot(X.T , (predictions - y))
            db = (1/num_samples) * np.sum(predictions - y)

            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db
            
    def predict(self , X):
        linear_model = np.dot(X , self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        class_pred = [0 if y <= 0.5 else 1 for y in y_pred]
        return class_pred



