import pickle, os
import numpy as np
from tqdm import tqdm
from data.generate import load_csv
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from torch.utils.data import DataLoader

model = SentenceTransformer('all-MiniLM-L6-v2')

class EmbeddingDataset:
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

def batch_encode_texts(texts, model, batch_size=32):
    dataset = EmbeddingDataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings = []

    for batch in tqdm(dataloader, desc="Batch Encoding", unit="batch"):
        batch_embeddings = model.encode(batch, convert_to_tensor=True)
        embeddings.append(batch_embeddings.cpu().numpy())

    return np.vstack(embeddings)

def extract_features(df, train=False):
    if train and os.path.exists('features.npy'):
        return np.load('features.npy')

    prompts = df["prompt"].tolist()
    responses_a = df["response_a"].tolist()
    responses_b = df["response_b"].tolist()

    # Batch embedding extraction
    prompt_embeddings = batch_encode_texts(prompts, model)
    response_a_embeddings = batch_encode_texts(responses_a, model)
    response_b_embeddings = batch_encode_texts(responses_b, model)

    # Sentiment analysis and verbosity
    sentiment_a = [TextBlob(text).sentiment.polarity for text in responses_a]
    sentiment_b = [TextBlob(text).sentiment.polarity for text in responses_b]
    verbosity_a = [len(text.split()) for text in responses_a]
    verbosity_b = [len(text.split()) for text in responses_b]

    # Compute cosine similarities
    similarity_a = np.diag(cosine_similarity(prompt_embeddings, response_a_embeddings))
    similarity_b = np.diag(cosine_similarity(prompt_embeddings, response_b_embeddings))
    similarity_a_b = np.diag(cosine_similarity(response_a_embeddings, response_b_embeddings))

    # Stack features
    features = np.column_stack([
        similarity_a, similarity_b, similarity_a_b,
        sentiment_a, sentiment_b,
        verbosity_a, verbosity_b
    ])
    if train:
        np.save("features.npy", features)

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
            early_stopping(stopping_rounds=50),
            log_evaluation(period=10)
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
