# OLS from Scratch

## Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Load data
data = pd.read_csv('assignment1_ols.csv')
data["Date"] = pd.to_datetime(data["Date"]).dt.date
print(data)

## Plot data
plt.plot(data["Date"], data["Close"])
plt.title("Date vs. Close Price")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.show()

## Convert Date to Ordinal Date
data["Date1"] = data["Date"].apply(lambda x: x.toordinal())
## Calculate OLS
X = data["Date1"].values.reshape(-1, 1)
y = data["Close"].values.reshape(-1, 1)

def OLS(X, y):
    X_bar = np.mean(X)
    y_bar = np.mean(y)
    X_y_bar = np.mean(X * y)
    X_sq_bar = np.mean(X ** 2)
    beta_1 = (X_y_bar - X_bar * y_bar) / (X_sq_bar - X_bar ** 2) #Slope equivalent to m or coefficients
    beta_0 = y_bar - beta_1 * X_bar #Intercept equivaalent to c
    return beta_0, beta_1

## Calculate OLS
beta_0, beta_1 = OLS(X, y)
print("constant", beta_0)
print("slope", beta_1)

## Plot OLS
plt.plot(data["Date"], data["Close"])
plt.plot(data["Date"], beta_0 + beta_1 * data["Date1"])
plt.title("OLS")
plt.xlabel("Date")
plt.ylabel("Close")
plt.show()
