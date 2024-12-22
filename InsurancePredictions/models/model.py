import pickle
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error

# Funzione obiettivo per Optuna
def objective(trial):
    # Carica i dati
    features = np.load("features.npy")
    results = np.load("results.npy")

    # Definizione degli iperparametri da ottimizzare
    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    max_depth = trial.suggest_int("max_depth", 5, 50)
    max_features = trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"])

    # Creazione del modello Random Forest
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        random_state=42,
    )

    # Valutazione del modello con cross-validation
    score = cross_val_score(model, features, results, cv=3, scoring="neg_mean_squared_error")
    return -score.mean()  # Optuna cerca di minimizzare il MSE

# Funzione per allenare il modello con i migliori parametri
def train_model():
    # Ottimizzazione con Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    # Mostra i migliori iperparametri
    print("Best Parameters:", study.best_params)

    # Allena il modello finale con i migliori parametri
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

    # Salva il modello addestrato
    with open("random_forest_model.pkl", "wb") as file:
        pickle.dump(model, file)

# Funzione per testare il modello e generare le previsioni
def test_model():
    with open("random_forest_model.pkl", "rb") as file:
        model: RandomForestRegressor = pickle.load(file)

    test_f = np.load("test_f.npy")
    predictions = model.predict(test_f)
    with open("predictions.csv", "w") as file:
        file.write("id,Premium Amount\n")
        for i, prediction in enumerate(predictions):
            file.write(f"{1200000 + i},{prediction}\n")
