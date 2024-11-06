import bentoml
from bentoml.io import JSON, NumpyNdarray

# Load the saved BentoML model
iris_model_runner = bentoml.sklearn.get("iris_classifier:latest").to_runner()

# Define the BentoML service
svc = bentoml.Service("iris_classifier_service", runners=[iris_model_runner])

@svc.api(input=NumpyNdarray(), output=JSON())
def predict(input_data):
    """API endpoint for making predictions."""
    prediction = iris_model_runner.predict.run(input_data)
    return {"prediction": prediction.tolist()}
