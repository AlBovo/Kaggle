import pickle, os
import numpy as np
from tqdm import tqdm
from data.generate import load_csv
from lightgbm import LGBMClassifier
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import lightgbm

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_features(df, train=False):
    if(train and os.path.exists('features.npy')):
        return np.load('features.npy')

    prompts = df["prompt"].tolist()
    responses_a = df["response_a"].tolist()
    responses_b = df["response_b"].tolist()

    prompt_embeddings = []
    response_a_embeddings = []
    response_b_embeddings = []

    for i in tqdm(range(len(prompts)), desc="Processing features", unit="rows"):
        prompt_embeddings.append(model.encode(prompts[i]))
        response_a_embeddings.append(model.encode(responses_a[i]))
        response_b_embeddings.append(model.encode(responses_b[i]))

    prompt_embeddings = np.array(prompt_embeddings)
    response_a_embeddings = np.array(response_a_embeddings)
    response_b_embeddings = np.array(response_b_embeddings)

    similarity_a = np.diag(cosine_similarity(prompt_embeddings, response_a_embeddings))
    similarity_b = np.diag(cosine_similarity(prompt_embeddings, response_b_embeddings))
    similarity_a_b = np.diag(cosine_similarity(response_a_embeddings, response_b_embeddings))

    features = np.column_stack([similarity_a, similarity_b, similarity_a_b])
    if(train): np.save("features.npy", features)
    return features

def train_model():
    train = load_csv()
    X = extract_features(train, True)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(train[["winner_model_a", "winner_model_b", "winner_tie"]].idxmax(axis=1))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        metric="multi_logloss",
        boosting_type="gbdt",
        random_state=42
    )
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=[
            lightgbm.early_stopping(stopping_rounds=50),
            lightgbm.log_evaluation(period=10)  # Log ogni 10 iterazioni
        ]
    )

    with open("model.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    y_proba = clf.predict_proba(X_val)
    log_loss_value = log_loss(y_val, y_proba)
    y_pred = np.argmax(y_proba, axis=1)
    print("Log Loss:", log_loss_value)
    print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))

def test_model():
    if not os.path.exists("model.pkl"):
        print("Model not found. Please train the model first.")
        return
    if not os.path.exists("label_encoder.pkl"):
        print("Label encoder not found. Please train the model first.")
        return
    test = load_csv(train=False)
    clf: LGBMClassifier = pickle.load(open("model.pkl", "rb"))
    label_encoder: LabelEncoder = pickle.load(open("label_encoder.pkl", "rb"))
    X_test = extract_features(test)
    test_predictions = clf.predict_proba(X_test)

    test["winner_model_a"] = test_predictions[:, 0]
    test["winner_model_b"] = test_predictions[:, 1]
    test["winner_tie"] = test_predictions[:, 2]

    submission = test[["id", "winner_model_a", "winner_model_b", "winner_tie"]]
    submission.columns = ["id", "winner_model_a", "winner_model_b", "winner_model_tie"]
    submission.to_csv("submission.csv", index=False)
