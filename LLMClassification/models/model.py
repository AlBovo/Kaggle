import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os

# Carica i dati
train = pd.read_csv(os.path.join("../data/", "train.csv"))
test = pd.read_csv(os.path.join("../data/", "test.csv"))

# Modello per ottenere embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Funzione per calcolare le similarità e le differenze tra i vettori
def extract_features(df):
    prompts = df["prompt"].tolist()
    responses_a = df["response_a"].tolist()
    responses_b = df["response_b"].tolist()

    # Inizializza liste per embeddings
    prompt_embeddings = []
    response_a_embeddings = []
    response_b_embeddings = []

    # Barra di avanzamento
    for i in tqdm(range(len(prompts)), desc="Processing features", unit="rows"):
        prompt_embeddings.append(model.encode(prompts[i]))
        response_a_embeddings.append(model.encode(responses_a[i]))
        response_b_embeddings.append(model.encode(responses_b[i]))

    # Converti le liste in array numpy
    prompt_embeddings = np.array(prompt_embeddings)
    response_a_embeddings = np.array(response_a_embeddings)
    response_b_embeddings = np.array(response_b_embeddings)

    # Similarità coseno tra il prompt e le risposte
    similarity_a = np.diag(cosine_similarity(prompt_embeddings, response_a_embeddings))
    similarity_b = np.diag(cosine_similarity(prompt_embeddings, response_b_embeddings))

    # Similarità coseno tra le due risposte
    similarity_a_b = np.diag(cosine_similarity(response_a_embeddings, response_b_embeddings))

    # Feature finale: combiniamo tutte le metriche
    features = np.column_stack([similarity_a, similarity_b, similarity_a_b])
    np.save("features.npy", features)
    return features

# Estrai le features
X = extract_features(train)

# Target (trasformazione delle etichette)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train[["winner_model_a", "winner_model_b", "winner_model_tie"]].idxmax(axis=1))

# Dividi il dataset in training e validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Addestra un modello
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Valutazione sul validation set
y_pred = clf.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))

def test_model():
    # Estrai features dal test set
    X_test = extract_features(test)

    # Prevedi sul test set
    test_predictions = clf.predict(X_test)
    test["winner_model"] = label_encoder.inverse_transform(test_predictions)

    # Salva la submission
    submission = test[["id", "winner_model"]]
    submission.columns = ["id", "winner_model_[a/b/tie]"]
    submission.to_csv("submission.csv", index=False)

