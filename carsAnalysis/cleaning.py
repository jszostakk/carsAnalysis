import numpy as np
import pandas as pd

#variables - year of production, power, transmission type, fuel type, mileage, fuel consumption

df = pd.read_csv('../archive/data.csv')

#Clean year values
df = df[df['year'].astype(str).str.len() == 4]
df = df[df['year'].astype(int) <= 2023]

#Clean power_kw
df = df[df['power_kw'].astype(int) <= 945]

#Clear power_ps
df = df[df['power_ps'].astype(int) <= 1900]

unique_years = df['year'].unique()

print('Wartosci year: ', unique_years)

print('typ: ', df['registration_date'].astype(str).str[-4:])

X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])
w = np.random.rand()
b = np.random.rand()

def hypothesis(x):
    return w * x + b

def mean_squared_error(y_pred, y_actual):
    return np.mean((y_pred - y_actual)**2)

rate = 0.01
epochs = 1000

for _ in range(epochs):
    y_pred = hypothesis(X)
    loss = mean_squared_error(y_pred, y)
    w -= rate * np.mean(2 * X * (y_pred - y))
    b -= rate * np.mean(2 * (y_pred - y))

prediction = hypothesis(6)
