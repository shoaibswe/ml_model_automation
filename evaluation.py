from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def evaluate(y_test, y_pred, model_name):
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Check if the problem is binary or multi-class
    if len(set(y_test)) == 2:
        auc = roc_auc_score(y_test, y_pred)
    else:
        auc = 'N/A'

    return {
        "model": model_name,
        "accuracy": accuracy,
        "f1_score": f1,
        "auc": auc,
    }
