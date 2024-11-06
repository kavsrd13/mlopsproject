import joblib
import bentoml

def load_model(filepath="../models/saved_model/iris_model.pkl"):
    """Load the trained model from disk."""
    return joblib.load(filepath)

def save_model_with_bentoml(model):
    """Save the model with BentoML for deployment."""
    bento_model = bentoml.sklearn.save_model(
        "iris_classifier",
        model,
        signatures={
            "predict": {"batchable": True}
        }
    )
    print(f"Model saved with BentoML: {bento_model}")

if __name__ == "__main__":
    # Load the trained model
    model = load_model()

    # Save the model with BentoML
    save_model_with_bentoml(model)
