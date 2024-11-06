import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load the data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

# Start MLFlow experiment
mlflow.start_run()

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Log the model
mlflow.sklearn.log_model(model, "iris_model")

# Log the performance metrics (e.g., accuracy)
accuracy = model.score(X_test, y_test)
mlflow.log_metric("accuracy", accuracy)

# End the MLFlow run
mlflow.end_run()
