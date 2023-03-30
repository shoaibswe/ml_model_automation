import sys
import argparse
import data_processing as dp
import models
import evaluation
import database as db

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train and test ML models with user-provided CSV data.")
    parser.add_argument("csv_file", help="Path to the CSV file containing the data.")
    parser.add_argument("models", nargs="+", help="List of ML models to use (e.g., RandomForest, LogisticRegression).")
    args = parser.parse_args()
    return args.csv_file, args.models

def main():
    # Parse command-line arguments (CSV file path, ML models, etc.)
    csv_file_path, ml_models = parse_arguments()

    # Load and preprocess data
    train_data, test_data = dp.load_and_preprocess_data(csv_file_path)

    # Train and test models
    results = []
    for model_name in ml_models:
        model = models.train_model(train_data, model_name)
        test_result = models.test_model(test_data, model)
        result = evaluation.evaluate(test_result, model_name)
        results.append(result)

    # Save results to the MySQL database
    db.save_results_to_database(results)

if __name__ == "__main__":
    main()
