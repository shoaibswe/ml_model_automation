import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_categorical_features(data, target_column):
    # Identify categorical columns
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    categorical_columns.remove(target_column)  # Remove the target column from the list

    # Use LabelEncoder to convert categorical columns to numeric values
    label_encoders = {}
    for column in categorical_columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    return data, label_encoders

def load_and_preprocess_data(csv_file_path, target_column):
    # Load the data from the CSV file
    data = pd.read_csv(csv_file_path)

    # Preprocess the data (e.g., handle missing values, convert categorical features, etc.)
    data, _ = preprocess_categorical_features(data, target_column)

    return data
