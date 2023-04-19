import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/dimalevin/Downloads/sample.csv')
#data.info() ---> no null values

learning_rates = [0.0001]
#learning_rates= [0.1]
batch_size = 20
epochs = 1000

def mse(y, y_pred):
    return np.mean((y - y_pred)**2)

def gradient_descent(data, learning_rate, epochs):
    a, b = 0, 0
    m = len(data)
    x = data['x'].values
    y = data['y'].values

    loss_history = []
    a_history = []
    b_history = []

    for epoch in range(epochs):
        y_pred = a * x + b
        loss = mse(y, y_pred)
        
        a_gradient = -2 * np.sum(x * (y - y_pred)) / m
        b_gradient = -2 * np.sum(y - y_pred) / m

        a = a - learning_rate * a_gradient
        b = b - learning_rate * b_gradient

        loss_history.append(loss)
        a_history.append(a)
        b_history.append(b)

    return loss_history, a_history, b_history

def stochastic_gradient_descent(data, learning_rate, epochs):
    a, b = 0, 0
    m = len(data)
    x = data['x'].values
    y = data['y'].values

    loss_history = []
    a_history = []
    b_history = []

    for epoch in range(epochs):
        for i in range(m):
            random_index = np.random.randint(0, m)
            xi = x[random_index]
            yi = y[random_index]
            y_pred = a * xi + b

            a_gradient = -2 * xi * (yi - y_pred)
            b_gradient = -2 * (yi - y_pred)

            a = a - learning_rate * a_gradient
            b = b - learning_rate * b_gradient

        y_pred_epoch = a * x + b
        loss = mse(y, y_pred_epoch)
        loss_history.append(loss)
        a_history.append(a)
        b_history.append(b)

    return loss_history, a_history, b_history

def mini_batch_gradient_descent(data, learning_rate, batch_size, epochs):
    a, b = 0, 0
    m = len(data)
    x = data['x'].values
    y = data['y'].values

    loss_history = []
    a_history = []
    b_history = []

    for epoch in range(epochs):
        shuffled_indices = np.random.permutation(m)
        x_shuffled = x[shuffled_indices]
        y_shuffled = y[shuffled_indices]

        for i in range(0, m, batch_size):
            xi = x_shuffled[i:i + batch_size]
            yi = y_shuffled[i:i + batch_size]
            y_pred = a * xi + b

            a_gradient = -2 * np.sum(xi * (yi - y_pred)) / batch_size
            b_gradient = -2 * np.sum(yi - y_pred) / batch_size

            a = a - learning_rate * a_gradient
            b = b - learning_rate * b_gradient

        y_pred_epoch = a * x + b
        loss = mse(y, y_pred_epoch)
        loss_history.append(loss)
        a_history.append(a)
        b_history.append(b)

    return loss_history, a_history, b_history

for lr in learning_rates:
    loss_history_gd, a_history_gd, b_history_gd = gradient_descent(data, lr, epochs)
    loss_history_sgd, a_history_sgd, b_history_sgd = stochastic_gradient_descent(data, lr, epochs)
    loss_history_mbgd, a_history_mbgd, b_history_mbgd = mini_batch_gradient_descent(data, lr, batch_size, epochs)
    
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].plot(loss_history_gd, label='GD')
axes[0].set_title('Loss vs. Epoch (Learning rate: {})'.format(lr))
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

axes[1].plot(a_history_gd, label='GD')
axes[1].set_title('Parameter a vs. Epoch (Learning rate: {})'.format(lr))
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Parameter a')
axes[1].legend()

axes[2].plot(b_history_gd, label='GD')
axes[2].set_title('Parameter b vs. Epoch (Learning rate: {})'.format(lr))
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Parameter b')
axes[2].legend()

plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].plot(loss_history_sgd, label='SGD')
axes[0].set_title('Loss vs. Epoch (Learning rate: {})'.format(lr))
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

axes[1].plot(a_history_sgd, label='SGD')
axes[1].set_title('Parameter a vs. Epoch (Learning rate: {})'.format(lr))
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Parameter a')
axes[1].legend()

axes[2].plot(b_history_sgd, label='SGD')
axes[2].set_title('Parameter b vs. Epoch (Learning rate: {})'.format(lr))
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Parameter b')
axes[2].legend()

plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].plot(loss_history_mbgd, label='MBGD')
axes[0].set_title('Loss vs. Epoch (Learning rate: {})'.format(lr))
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

axes[1].plot(a_history_mbgd, label='MBGD')
axes[1].set_title('Parameter a vs. Epoch (Learning rate: {})'.format(lr))
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Parameter a')
axes[1].legend()

axes[2].plot(b_history_mbgd, label='MBGD')
axes[2].set_title('Parameter b vs. Epoch (Learning rate: {})'.format(lr))
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Parameter b')
axes[2].legend()

plt.show()




