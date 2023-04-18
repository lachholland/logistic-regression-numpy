import numpy as np

def null_fix(dataset, column, fill):
    for i in range(dataset.shape[0]):
        if not isinstance(dataset[i, column], str):
            dataset[i, column] = fill
            
def one_hot_encoding(data, column, encoding):
   one_hot = np.zeros((len(data), len(encoding)))
   for i in range(len(data)):
      one_hot[i, encoding[data[i, column]]] = 1
   encoded = np.concatenate((data, one_hot), axis=1)
   return np.delete(encoded, column, 1)

def sigmoid(z):
    return  1.0 / (1.0 + np.exp(-z))

def predict(X, w, b):
    z = np.dot(X, w) + b
    y_pred = sigmoid(z)
    return y_pred

def log_loss(y, y_hat):
    eps = 1e-15
    J = -np.mean(y * np.log(y_hat + eps) - (1 - y) * np.log(1 - y_hat + eps))
    return J

def gradients(X, y, y_hat):
    m = X.shape[0]
    dw = (1/m) * np.dot(X.T, (y_hat - y))
    db = (1/m) * np.sum(y_hat - y)
    return dw, db