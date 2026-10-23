
import joblib
import numpy as np
from pathlib import Path


def test_model_loads():
    assert Path("model.pkl").exists()
    model = joblib.load("model.pkl")
    assert model is not None


def test_prediction_shape():
    model = joblib.load("model.pkl")
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    predictions = model.predict(sample)
    assert predictions.shape == (1,)
