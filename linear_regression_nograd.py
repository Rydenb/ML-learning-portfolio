import pandas as pd
import matplotlib.pyplot as plt


data=pd.read_csv("linear_regression_test_data.csv")
df = pd.DataFrame(data)
length = len(data)
x = df["x_values"].tolist()
y = df["y_values"].tolist()

def least_squares(xi, yi):
    x = sum(xi)/len(xi)
    y = sum(yi)/len(yi)
    sum1 = 0
    sum2 = 0

    for i in range(length):
        sum1 += ((x-xi[i])*(y-yi[i]))
        sum2+=(x-xi[i])**2

    m = sum1/sum2
    c = y-m*x

    return m, c

m = least_squares(x, y)[0]
c = least_squares(x, y)[1]

print("m: " + str(m) + "c: " + str(c))

plt.scatter(x, y)
plt.axline((0, c), slope=m, color = "red", label = 'my line')
plt.show()
