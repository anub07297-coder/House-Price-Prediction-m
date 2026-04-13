# House Price Prediction Model

This project provides a clean, beginner-friendly scaffold for training a housing price prediction model using `housing.csv`.

## Project Structure

- `data/`
  - `housing.csv` — dataset file (assumed to already exist)
- `src/`
  - `data_loader.py` — load and split the dataset
  - `preprocessing.py` — handle missing values, encoding, and scaling
  - `train.py` — train and save the regression model
  - `evaluate.py` — calculate RMSE and R² metrics
  - `predict.py` — load saved model and make predictions
- `models/`
  - `saved_model.pkl` — placeholder for the saved trained model
- `notebooks/`
  - `EDA.ipynb` — exploratory data analysis notebook
- `main.py` — main CLI entrypoint
- `requirements.txt` — dependency list

## Installation

1. Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Train the model

```bash
python main.py train --data data/housing.csv --output models/saved_model.pkl
```

You can choose the model type:

```bash
python main.py train --data data/housing.csv --model random_forest
python main.py train --data data/housing.csv --model linear_regression
```

### Evaluate the saved model

```bash
python main.py evaluate --data data/housing.csv --model-path models/saved_model.pkl
```

### Predict with new data

Using a CSV file containing the same feature columns:

```bash
python main.py predict --model-path models/saved_model.pkl --input-file data/sample_input.csv
```

Using a JSON string for a single row:

```bash
python main.py predict --model-path models/saved_model.pkl --json-input '{"longitude": -122.23, "latitude": 37.88, "housing_median_age": 41, "total_rooms": 880, "total_bedrooms": 129, "population": 322, "households": 126, "median_income": 8.3252, "ocean_proximity": "NEAR BAY"}'
```

## Notes

- The model pipeline includes missing value handling, categorical encoding, and scaling.
- Evaluation reports RMSE and R² to assess regression performance.
- The saved model file contains the trained estimator, scaler, and feature layout for future predictions.
