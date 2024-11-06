import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# Load dataset function
def load_data(filepath):
    """Load the iris dataset from the specified file path."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found.")
    return pd.read_csv(filepath)

# Preprocess the dataset
def preprocess_data(data):
    """Preprocess the data by encoding labels and scaling features."""
    # Encode the 'species' column to numeric
    label_encoder = LabelEncoder()
    data['species'] = label_encoder.fit_transform(data['species'])

    # Split data into features and target
    X = data.drop('species', axis=1)
    y = data['species']

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, label_encoder

if __name__ == "__main__":
    # Specify the file path
    filepath = "../data/raw/iris.csv"

    # Load and preprocess the data
    iris_data = load_data(filepath)
    X_train, X_test, y_train, y_test, encoder = preprocess_data(iris_data)

    # Save the processed data to the processed directory
    pd.DataFrame(X_train).to_csv("../data/processed/X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv("../data/processed/X_test.csv", index=False)
    pd.DataFrame(y_train).to_csv("../data/processed/y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv("../data/processed/y_test.csv", index=False)

    print("Data preprocessing complete. Files saved in data/processed/")
