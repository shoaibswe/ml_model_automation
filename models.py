from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC

def train_model(train_data, model_name, target_column):
    if model_name == "Logistic Regression":
        model = LogisticRegression(random_state=42)
    elif model_name == "Decision Tree Classifier":
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == "Random Forest Classifier":
        model = RandomForestClassifier(random_state=42)
    elif model_name == "Gradient Boosting Classifier":
        model = GradientBoostingClassifier(random_state=42)
    elif model_name == "Support Vector Machine Classifier":
        model = SVC(random_state=42)
    elif model_name == "AdaBoost Classifier":
        model = AdaBoostClassifier(random_state=42)

    X_train = train_data.drop(target_column, axis=1)
    y_train = train_data[target_column]
    model.fit(X_train, y_train)
    return model

def test_model(test_data, model, target_column):
    X_test = test_data.drop(target_column, axis=1)
    y_test = test_data[target_column]
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)
    except AttributeError:
        y_proba = None
    return y_test, y_pred, y_proba
