import numpy as np
import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import pickle

def train_model():
    features = np.load("features.npy")
    results = np.load("results.npy")
    
    model = XGBRegressor(
        n_estimators=189,
        max_depth=32,
        learning_rate=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        num_parallel_tree=100,
        booster="gbtree",
        tree_method="hist",
        device="gpu",
        random_state=42
    )

    model.fit(features, results)

    with open("xgboost_model.pkl", "wb") as file:
        pickle.dump(model, file)


def test_model():
    with open("random_forest_model.pkl", "rb") as file:
        model: XGBRegressor = pickle.load(file)

    test_f = np.load("test_f.npy")
    predictions = model.predict(test_f)
    with open("predictions.csv", "w") as file:
        file.write("id,Premium Amount\n")
        for i, prediction in enumerate(predictions):
            file.write(f"{1200000 + i},{prediction}\n")
