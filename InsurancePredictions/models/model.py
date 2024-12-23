import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle

def train_model():
    features = np.load("features.npy")
    results = np.load("results.npy")
    
    model = RandomForestRegressor(
        n_estimators=386,
        max_depth=7,
        max_features="log2",
        random_state=42,
    )

    model.fit(features, results)

    with open("random_forest_model.pkl", "wb") as file:
        pickle.dump(model, file)


def test_model():
    with open("random_forest_model.pkl", "rb") as file:
        model: RandomForestRegressor = pickle.load(file)

    test_f = np.load("test_f.npy")
    predictions = model.predict(test_f)
    with open("predictions.csv", "w") as file:
        file.write("id,Premium Amount\n")
        for i, prediction in enumerate(predictions):
            file.write(f"{1200000 + i},{prediction}\n")
