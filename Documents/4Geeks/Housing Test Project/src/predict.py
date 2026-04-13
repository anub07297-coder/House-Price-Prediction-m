import pandas as pd
import joblib
import numpy as np
import os

# Load model
model_path = os.path.join('..', 'models', 'housing_model.pkl')
model = joblib.load(model_path)

# Load data for prediction (using test data as example)
data_path = os.path.join('..', 'data', 'housing.csv')
df = pd.read_csv(data_path)
X = df.iloc[:, :-1]

# Predict on first few samples
predictions = model.predict(X.head())
print('Predictions for first 5 samples:')
for i, pred in enumerate(predictions):
    print(f'Sample {i+1}: {pred}')
