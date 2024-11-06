import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import joblib

def load_test_data():
    """Load the preprocessed test data."""
    X_test = pd.read_csv("../data/processed/X_test.csv")
    y_test = pd.read_csv("../data/processed/y_test.csv").values.ravel()  # Flatten the array
    return X_test, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluate the model using the test data."""
    y_pred = model.predict(X_test)
    print("Model Evaluation:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

if __name__ == "__main__":
    # Load test data
    X_test, y_test = load_test_data()

    # Load the trained model
    model = joblib.load("../models/saved_model/iris_model.pkl")

    # Evaluate the model
    evaluate_model(model, X_test, y_test)
