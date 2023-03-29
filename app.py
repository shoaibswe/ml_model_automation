from flask import Flask, render_template, request
import data_processing as dp
import models
import evaluation
import os
import tempfile
from sklearn.model_selection import train_test_split

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Save the uploaded CSV file
        csv_file = request.files["csv_file"]
        csv_file_path = os.path.join(tempfile.mkdtemp(), "data.csv")
        csv_file.save(csv_file_path)

        # Get the selected ML models
        ml_models = request.form.getlist("models")

        # Get the target column name from the form
        target_column = request.form["target"]

        # Pass the target column to the load_and_preprocess_data function
        data = dp.load_and_preprocess_data(csv_file_path, target_column)

        # Split the data into training and testing sets
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

        # Train and test models
        results = []
        for model_name in ml_models:
            model = models.train_model(train_data, model_name, target_column)
            y_test, y_pred, y_proba = models.test_model(test_data, model, target_column)
            result = evaluation.evaluate(y_test, y_pred, model_name)
            results.append(result)

        return render_template("results.html", results=results)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
