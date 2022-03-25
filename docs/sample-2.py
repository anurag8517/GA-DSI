import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
# from sklearn.cross_val_score import cross_val_score --- this is not a module
from sklearn.model_selection import cross_val_score

# Load data
data = pd.read_csv('data/part-2-data.train.csv')


# Setup data for prediction
y = data.SalaryNormalized
X = pd.get_dummies(data.ContractType)

# Setup model
model = LinearRegression()

# Evaluate model
# scores = cross_val_score(model, X, y, cv=5, scoring='mean_absolute_error') --- the full name of this metric is below if passing as a string
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
print(scores.mean())