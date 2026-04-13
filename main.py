import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from src.data_loader import load_data, split_data
from src.evaluate import evaluate_model
from src.predict import load_saved_model, predict_new_data
from src.preprocessing import preprocess_data
from src.train import save_model, train_model


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def train(args):
    df = load_data(args.data)
    X_train, X_test, y_train, y_test = split_data(df, target_col=args.target)
    X_train_prepared, X_test_prepared, scaler, feature_columns = preprocess_data(X_train, X_test)
    model = train_model(X_train_prepared, y_train, model_type=args.model)
    save_model(model, scaler, feature_columns, args.output)
    logging.info("Training complete. Saved model to %s", args.output)
    if args.evaluate:
        metrics = evaluate_model(model, X_test_prepared, y_test)
        logging.info("Evaluation results: %s", metrics)


def evaluate(args):
    df = load_data(args.data)
    X_train, X_test, y_train, y_test = split_data(df, target_col=args.target)
    model, scaler, feature_columns = load_saved_model(args.model_path)
    X_train_prepared, X_test_prepared, _, _ = preprocess_data(X_train, X_test)
    metrics = evaluate_model(model, X_test_prepared, y_test)
    print("Evaluation results:")
    print(json.dumps(metrics, indent=2))


def predict(args):
    model, scaler, feature_columns = load_saved_model(args.model_path)

    if args.input_file:
        new_data = pd.read_csv(args.input_file)
    elif args.json_input:
        new_data = pd.DataFrame([json.loads(args.json_input)])
    else:
        raise ValueError("Please provide --input-file or --json-input for prediction")

    predictions = predict_new_data(model, scaler, new_data, feature_columns)
    results = pd.DataFrame({"prediction": predictions})
    print(results.to_string(index=False))


def create_parser():
    parser = argparse.ArgumentParser(
        description="House Price Prediction CLI",
        epilog=(
            "Examples:\n"
            "  python main.py train --data data/housing.csv --output models/saved_model.pkl\n"
            "  python main.py evaluate --data data/housing.csv --model-path models/saved_model.pkl\n"
            "  python main.py predict --model-path models/saved_model.pkl --input-file data/sample_input.csv"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a regression model")
    train_parser.add_argument("--data", required=True, help="Path to housing CSV file")
    train_parser.add_argument("--output", default="models/saved_model.pkl", help="Path to save the trained model")
    train_parser.add_argument("--model", choices=["linear_regression", "random_forest"], default="random_forest")
    train_parser.add_argument("--target", default="median_house_value", help="Name of the target column")
    train_parser.add_argument("--evaluate", action="store_true", help="Run evaluation after training")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a saved model")
    eval_parser.add_argument("--data", required=True, help="Path to housing CSV file")
    eval_parser.add_argument("--model-path", default="models/saved_model.pkl", help="Path to the saved model")
    eval_parser.add_argument("--target", default="median_house_value", help="Name of the target column")

    predict_parser = subparsers.add_parser("predict", help="Predict house prices for new data")
    predict_parser.add_argument("--model-path", default="models/saved_model.pkl", help="Path to the saved model")
    predict_parser.add_argument("--input-file", help="CSV file with new feature rows")
    predict_parser.add_argument("--json-input", help="JSON string with feature values for a single row")

    return parser


def main():
    configure_logging()
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "predict":
        predict(args)


if __name__ == "__main__":
    main()
