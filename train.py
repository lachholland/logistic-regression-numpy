import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import null_fix, one_hot_encoding, sigmoid, predict, log_loss, gradients

def train(X, y, lr, epochs, bs=30):
    m, n = X.shape
    w = [-1.47858625, 1.14375404, 0.33849641, -0.41528791, 0.63278187, 2.27069286, 0.18186626, 0.24822059] 
    b = 0
    loss_history = []
    X = X.astype(np.float64)
    for epoch in range(epochs):
        for i in range((m-1)//bs + 1):
            b_start = i * bs
            b_end = b_start + bs
            xb = X[b_start:b_end]
            yb = y[b_start:b_end]
            y_hat = predict(xb, w, b)
            dw, db = gradients(xb, yb, y_hat)
            w -= lr*dw
            b -= lr*db
        loss = log_loss(y, sigmoid(np.dot(X, w) + b))
        loss_history.append(loss)
    return loss_history

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
y_train = train_data['Survived'].copy()
train_data.drop(['PassengerId','Name', 'Survived','Age', 'Ticket', 'Fare', 'Cabin'],inplace=True, axis=1)

x_train_numpy = train_data.values
x_test_numpy = test_data.values
null_fix(x_train_numpy, 4, 'S')
null_fix(x_test_numpy, 10, 'S')

embarked_encoding = {'S': 0, 'C': 1, 'Q': 2}
sex_encoding = {'male': 1, 'female': 0}
x_train_encoded1 = one_hot_encoding(x_train_numpy, 4, embarked_encoding)
x_train_encoded2 = one_hot_encoding(x_train_encoded1, 1, sex_encoding)
y_train_numpy = y_train.values

result = train(x_train_encoded2, y_train_numpy, 0.01, 100, 30)
epochs = [i + 1 for i in range(len(result))]
plt.plot(epochs, result)
plt.title("Training Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()