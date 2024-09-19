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
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary and predict the Regression value
   
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: AARON I
RegisterNumber:  212223230002
*/
import pandas as pd
df = pd.read_csv("/content/Placement_Data.csv")
print(df.head())
df1 = df.copy()
df1 = df1.drop(['sl_no', 'salary'], axis = 1)
# print(df1.head())
print(df1.isnull().sum())
print(df1.duplicated().sum())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df1['gender'] = le.fit_transform(df1['gender'])
df1['ssc_b'] = le.fit_transform(df1['ssc_b'])
df1['ssc_p'] = le.fit_transform(df1['ssc_p'])
df1['hsc_b'] = le.fit_transform(df1['hsc_b'])
df1['hsc_p'] = le.fit_transform(df1['hsc_p'])
df1['hsc_s'] = le.fit_transform(df1['hsc_s'])
df1['degree_t'] = le.fit_transform(df1['degree_t'])
df1['workex'] = le.fit_transform(df1['workex'])
df1['status'] = le.fit_transform(df1['status'])
df1["specialisation"]=le.fit_transform(df1["specialisation"])

print(df1)

x = df1.iloc[:,:-1]
# print(x)

y = df1['status']
# print(y)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
x = imputer.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print(y_pred)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test, y_pred)
print(classification_report1)

print(lr.predict([[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]]))
```

## Output:
![Screenshot 2024-09-19 114025](https://github.com/user-attachments/assets/e6132dda-82a4-45d6-bef9-fdd304cefa5e)

![Screenshot 2024-09-19 114038](https://github.com/user-attachments/assets/b4212c7d-c4bd-4444-a817-5453d17c40fd)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

