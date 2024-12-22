import pickle
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

def objective(trial):
    features = np.load("features.npy")
    results = np.load("results.npy")

    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    max_depth = trial.suggest_int("max_depth", 5, 50)
    max_features = trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"])

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        random_state=42,
    )

    score = cross_val_score(model, features, results, cv=3, scoring="neg_mean_squared_error")
    return -score.mean()

def train_model():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print("Best Parameters:", study.best_params)

    features = np.load("features.npy")
    results = np.load("results.npy")
    best_params = study.best_params
    model = RandomForestRegressor(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        max_features=best_params["max_features"],
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
