import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv', header=None)
data.columns = ['x', 'y', 'color']
data = data.drop(columns=["color"])

print(data)

def loss(m, b, points):
    total_error = 0
    for i in range(len(points)):
        X = points.iloc[i].x
        Y = points.iloc[i].y
        total_error += (Y-(m*X+b))**2
    total_error/float(len(points))

def gradient_descent(m_curr, b_curr, points, lr):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        X = points.iloc[i].x
        Y = points.iloc[i].y

        error = Y - (m_curr * X + b_curr)

        m_gradient += -(2/n) * X * error
        b_gradient += -(2/n) * error
    m = m_curr - m_gradient * lr
    b = b_curr - b_gradient * lr

    return m, b

m = 0
b = 0
lr = 0.0001
epochs = 100

for i in range(epochs):
    if i % 10 == 0:
        print("epoch: " + str(i))
    m, b = gradient_descent(m, b, data, lr)

print(m, b)

plt.scatter(data.x, data.y, color="blue")
plt.plot(list(range(-1, 1)), [m*x+b for x in range(-1,1)], color="red")
plt.show()
