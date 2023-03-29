from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def train_model(train_data, model_name, target_column):
    # Extract features and target from the training data
    X_train = train_data.drop(target_column, axis=1)
    y_train = train_data[target_column]

    # Train the model based on the model_name
    if model_name == "RandomForest":
        model = RandomForestClassifier(random_state=42)
    elif model_name == "LogisticRegression":
        model = LogisticRegression(random_state=42)
    # Add more models as needed

    model.fit(X_train, y_train)
    return model

def test_model(test_data, model, target_column):
    X_test = test_data.drop(target_column, axis=1)
    y_test = test_data[target_column]

    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return y_test, y_pred, y_proba
