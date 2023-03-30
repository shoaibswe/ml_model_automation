from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate(y_test, y_pred, model_name):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    try:
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
    except NameError:
        auc = "N/A"
    
    return {"model_name": model_name, "accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1, "auc_score": auc}
