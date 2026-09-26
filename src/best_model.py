"""Train, evaluate, and save the tuned abstract classifier."""

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from src.text_preprocessing import ensure_nltk_resources, preprocess_abstracts


def model_scores(name, truth, prediction):
    return {
        "Model": name,
        "Accuracy": accuracy_score(truth, prediction),
        "Macro F1": f1_score(truth, prediction, average="macro"),
        "Balanced accuracy": balanced_accuracy_score(truth, prediction),
        "AI recall": recall_score(truth, prediction, pos_label="ai"),
    }


def main():
    ensure_nltk_resources()
    with (PROJECT_DIR / "Dataset" / "research_abstracts.json").open(encoding="utf-8") as file:
        df = pd.DataFrame(json.load(file))[["Abstract", "Label"]]
    df = df.drop_duplicates().dropna()

    text_train, text_test, y_train, y_test = train_test_split(
        df["Abstract"], df["Label"], test_size=0.20, random_state=42
    )

    pipeline = Pipeline([
        ("preprocess", FunctionTransformer(preprocess_abstracts, validate=False)),
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("classifier", MultinomialNB()),
    ])
    search = GridSearchCV(
        pipeline,
        param_grid={
            "tfidf__max_features": [500, 2000, 5000],
            "classifier__alpha": [0.05, 0.1, 1.0],
        },
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
    )
    search.fit(text_train, y_train)

    predictions = search.predict(text_test)
    print(pd.Series(model_scores("Tuned Naive Bayes", y_test, predictions)))
    print("Best training-CV parameters:", search.best_params_)
    print("Best training-CV macro F1:", round(search.best_score_, 4))

    model_path = PROJECT_DIR / "model" / "model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as file:
        pickle.dump(search.best_estimator_, file)

    with model_path.open("rb") as file:
        loaded_model = pickle.load(file)
    assert (loaded_model.predict(text_test) == predictions).all()
    print("Saved model:", model_path)


if __name__ == "__main__":
    main()
