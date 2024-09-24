# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary python packages
2. Read the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.'
5. Define a function to plot the decision boundary and predict the Regression value.


## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Aaron I
RegisterNumber:  212223230002
*/
import pandas as pd
import numpy as np

df = pd.read_csv("/content/Placement_Data.csv")
# print(df)

df = df.drop('sl_no',axis = 1)
df = df.drop('salary',axis = 1)

# print(df)

df['gender'] = df['gender'].astype('category')
df['ssc_b'] = df['ssc_b'].astype('category')
df['hsc_b'] = df['hsc_b'].astype('category')
df['degree_t'] = df['degree_t'].astype('category')
df['workex'] = df['workex'].astype('category')
df['specialisation'] = df['specialisation'].astype('category')
df['status'] = df['status'].astype('category')
df['hsc_s'] = df['hsc_s'].astype('category')
print(df.dtypes)
# print(df)

X=df.iloc[:,: -1]
Y=df["status"]

df['gender'] = df['gender'].cat.codes
df['ssc_b'] = df['ssc_b'].cat.codes
df['hsc_b'] = df['hsc_b'].cat.codes
df['degree_t'] = df['degree_t'].cat.codes
df['workex'] = df['workex'].cat.codes
df['specialisation'] = df['specialisation'].cat.codes
df['status'] = df['status'].cat.codes
df['hsc_s'] = df['hsc_s'].cat.codes

print(df)

X = df.iloc[:,:-1].values
Y=df['status'].values
print(X)
print(Y)

theta = np.random.randn(X.shape[1])
y = Y

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1-h))

def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta = theta - alpha * gradient
    return theta

theta = gradient_descent(theta, X, y, alpha = 0.01, num_iterations = 1000)

def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5, 1, 0)
    return y_pred

y_pred = predict(theta, X)

accuracy = np.mean(y_pred.flatten() == y)
print("Accury:",accuracy)

print("Y predicted :")
print(y_pred)

print("Y :")
print(y)

x_new = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
y_pred_new1 = predict(theta, x_new)
print("Y Predict new 1:",y_pred_new1)

x_new = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])
y_pred_new2 = predict(theta, x_new)
print("Y Predict new 2:", y_pred_new2)
```

## Output:

![Screenshot 2024-09-24 202233](https://github.com/user-attachments/assets/40c0d243-df1d-4603-bb3c-25774b7a0558)

![Screenshot 2024-09-24 202246](https://github.com/user-attachments/assets/6b288079-d51e-4cfa-b532-c6d3df8ae9b7)

![Screenshot 2024-09-24 202256](https://github.com/user-attachments/assets/d37c76ba-a0b8-4c5f-a514-598be044f498)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

