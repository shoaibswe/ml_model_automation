import pandas as pd

def load_and_preprocess_data(csv_file_path, target_column):
    try:
        # Load the CSV file
        df = pd.read_csv(csv_file_path)

        # Check if the target column exists in the data
        if target_column not in df.columns:
            raise ValueError("Target column not found in data")

        # Split the data into features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # One-hot encode categorical features
        X = pd.get_dummies(X)

        # Fill missing values with the mean of the column
        X = X.fillna(X.mean())

        # Return the preprocessed data
        return pd.concat([X, y], axis=1)

    except Exception as e:
        print("Error processing data:", e)
        return "An error occurred while processing the data. Please check if the CSV file and target column are correctly specified and try again."
