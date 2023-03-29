from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def train_model(train_data, model_name, target_column):
    try:
        X_train = train_data.drop(columns=[target_column])
        y_train = train_data[target_column]

        if model_name == "LogisticRegression":
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
        elif model_name == "RandomForestClassifier":
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
        elif model_name == "SVM":
            model = SVC(probability=True)
            model.fit(X_train, y_train)
        else:
            raise ValueError("Invalid model name")

        return model

    except Exception as e:
        print("Error training model:", e)
        return "An error occurred while training the model. Please check if the training data and model name are correctly specified and try again."

def test_model(test_data, model, target_column):
    try:
        X_test = test_data.drop(columns=[target_column])
        y_test = test_data[target_column]
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        return y_test, y_pred, y_proba

    except Exception as e:
        print("Error testing model:", e)
        return "An error occurred while testing the model. Please check if the testing data and model are correctly specified and try again."
