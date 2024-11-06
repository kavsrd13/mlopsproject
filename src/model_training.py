import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

def load_processed_data():
    """Load the preprocessed training and test data."""
    X_train = pd.read_csv("../data/processed/X_train.csv")
    y_train = pd.read_csv("../data/processed/y_train.csv").values.ravel()  # Flatten the array
    return X_train, y_train

def train_model(X_train, y_train):
    """Train a logistic regression model."""
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model

def save_model(model, filepath="../models/saved_model/iris_model.pkl"):
    """Save the trained model to disk."""
    joblib.dump(model, filepath)
    print(f"Model saved at {filepath}")

if __name__ == "__main__":
    # Load processed data
    X_train, y_train = load_processed_data()

    # Train the model
    model = train_model(X_train, y_train)

    # Save the model
    save_model(model)
