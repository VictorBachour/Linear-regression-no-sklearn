import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv('data1.csv')
# original graph
plt.figure(1)
plt.scatter(data.iloc[:,0], data.iloc[:,1])
plt.xlabel(data.columns[0])
plt.ylabel(data.columns[1])
plt.title(f"{data.columns[1]} vs {data.columns[0]}")


def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i, 0]
        y = points.iloc[i, 1]
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))

def gradient_descent(m_curr, b_curr, points, lr):
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i, 0]
        y = points.iloc[i, 1]

        m_gradient += -(2/n) * x * (y - (m_curr * x + b_curr))
        b_gradient += -(2/n) * (y - (m_curr * x + b_curr))

    m = m_curr - m_gradient * lr
    b = b_curr - b_gradient * lr

    return m,b


m = 0
b = 0
lr = .00000001
epochs = 300

for i in range(epochs):
    m, b = gradient_descent(m, b, data, lr)
print(m, b)
plt.figure(2)
plt.scatter(data.iloc[:, 0], data.iloc[:, 1], color='red')
x_vals = np.linspace(min(data.iloc[:, 0]), max(data.iloc[:, 0]), 100)
y_vals = m * x_vals + b
plt.plot(x_vals, y_vals, color='blue')
plt.xlabel(data.columns[0])
plt.ylabel(data.columns[1])
plt.title(f"{data.columns[1]} vs {data.columns[0]}")
plt.show()