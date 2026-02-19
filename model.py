from sklearn.ensemble import RandomForestRegressor
import numpy as np
import joblib

MODEL_PATH = "param_model.pkl"

def train_dummy_model():
    X = np.random.rand(30, 11)

    Y = np.array([
        [2.0, 0.8, 1.2, 1.0, 0.8],
        [2.4, 0.9, 1.3, 1.1, 0.9],
        [1.8, 0.7, 1.1, 0.9, 0.7]
    ] * 10)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, Y)
    joblib.dump(model, MODEL_PATH)

def load_model():
    return joblib.load(MODEL_PATH)
