import pickle, os
import numpy as np
from tqdm import tqdm
from data.generate import load_csv
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_features(df):
    if os.path.exists("features.npy"):
        return np.load("features.npy")
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
    np.save("features.npy", features)
    return features

def train_model():
    train = load_csv()
    X = extract_features(train)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(train[["winner_model_a", "winner_model_b", "winner_tie"]].idxmax(axis=1))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42
    )
    clf.fit(X_train, y_train)

    with open("model.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    y_proba = clf.predict_proba(X_val)
    y_pred = np.argmax(y_proba, axis=1)
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))

def test_model():
    if not os.path.exists("model.pkl"):
        print("Model not found. Please train the model first.")
        return
    if not os.path.exists("label_encoder.pkl"):
        print("Label encoder not found. Please train the model first.")
        return
    test = load_csv(train=False)
    clf : XGBClassifier = pickle.load(open("model.pkl", "rb"))
    label_encoder : LabelEncoder = pickle.load(open("label_encoder.pkl", "rb"))
    X_test = extract_features(load_csv(train=False))
    test_predictions = clf.predict(X_test)
    test["winner_model"] = label_encoder.inverse_transform(test_predictions)

    submission = test[["id", "winner_model"]]
    submission.columns = ["id", "winner_model_[a/b/tie]"]
    submission.to_csv("submission.csv", index=False)

