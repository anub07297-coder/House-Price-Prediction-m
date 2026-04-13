import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import os

# Load data
data_path = os.path.join('..', 'data', 'housing.csv')
df = pd.read_csv(data_path)

# Assume the last column is the target 'price'
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Save model
model_path = os.path.join('..', 'models', 'housing_model.pkl')
joblib.dump(model, model_path)
print(f'Model saved to {model_path}')
