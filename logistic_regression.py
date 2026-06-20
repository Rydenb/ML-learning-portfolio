import pandas as pd
import numpy as np

df1 = pd.read_csv("train.csv")
df2 = pd.read_csv("test.csv")

df1_clean = df1.drop(columns=['Name', 'Ticket', 'Cabin'], errors='ignore')
df1_clean['Sex'] = df1_clean['Sex'].map({'male': 1, 'female': 0})
df1_clean['Age'] = df1_clean['Age'].fillna(df1_clean['Age'].median())
df1_clean = pd.get_dummies(df1_clean, columns=['Embarked'], drop_first=True)


df2_clean = df2.drop(columns=['Name', 'Ticket', 'Cabin'], errors='ignore')
df2_clean['Sex'] = df2_clean['Sex'].map({'male': 1, 'female': 0})
df2_clean['Age'] = df2_clean['Age'].fillna(df1_clean['Age'].median())
df2_clean = pd.get_dummies(df2_clean, columns=['Embarked'], drop_first=True)

results1 = df1_clean['Survived'].values.astype(float)
features1 = df1_clean.drop(columns=['Survived', 'PassengerId'], errors='ignore').values.astype(float)
features2 = df2_clean.drop(columns=['PassengerId'], errors='ignore').values.astype(float)


def sig(z):
    return 1 / (1+np.exp(-1*z))

def calculate_grad(w, X, y):
    return (X.T @ (sig(X @ w) - y)) / y.size

def gradient_descent(X, y, alpha, epochs=1000):
    w = np.zeros(X.shape[1])
    for i in range(epochs):
        grad = calculate_grad(w, X, y)
        w -= alpha*grad
    return w

weights = gradient_descent(features1, results1, 0.01)


def predict(weights, test, threshold=0.5):
    logits = features2 @ weights
    probabilities = sig(logits)

    predictions = (probabilities >= threshold).astype(int)

    return predictions

predictions = predict(weights, features2)

output_df = pd.DataFrame({
    'PassengerId': df2['PassengerId'],
    'Survived': predictions
})

output_df.to_csv('predictions.csv', index=False)

print("predictions saved")
