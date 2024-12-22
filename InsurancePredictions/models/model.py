import numpy as np
import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import pickle

def objective(trial, features, results):
    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    max_depth = trial.suggest_int("max_depth", 5, 50)
    
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        tree_method="gpu_hist",  # Enable GPU acceleration
        random_state=42,
    )

    score = cross_val_score(model, features, results, cv=3, scoring="neg_mean_squared_error")
    return -score.mean()

def train_model():
    features = np.load("features.npy")
    results = np.load("results.npy")
    
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(), 
        pruner=MedianPruner()
    )
    study.optimize(
        lambda trial: objective(trial, features, results), 
        n_trials=50,
        show_progress_bar=True
    )

    print("Best Parameters:", study.best_params)

    best_params = study.best_params
    model = XGBRegressor(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        tree_method="gpu_hist",
        random_state=42,
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
