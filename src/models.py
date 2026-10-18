import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NEWS = ["sentiment_mean", "sentiment_std", "positive_share", "negative_share", "article_count"]
MARKET = ["return_1d", "return_5d", "volatility_5d"]

def chronological_split(frame, test_fraction=0.20):
    dates = sorted(frame.date.dropna().unique())
    if len(dates) < 8:
        raise ValueError("Need at least 8 unique dates for chronological evaluation.")
    cut = max(1, int(len(dates)*(1-test_fraction)))
    train, test = frame[frame.date.isin(dates[:cut])].copy(), frame[frame.date.isin(dates[cut:])].copy()
    if train.target_up.nunique() < 2:
        raise ValueError("Training split has only one target class.")
    return train, test

def structured_model():
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))])

def fit_compare_models(frame):
    train, test = chronological_split(frame)
    y_train, y_test = train.target_up, test.target_up
    pred = test[["date","ticker","target_up","next_return"]].copy()
    models = {}
    for name, cols in {"market_only": MARKET, "sentiment_only": NEWS, "news_plus_market": NEWS+MARKET}.items():
        model = structured_model().fit(train[cols], y_train)
        prob = model.predict_proba(test[cols])[:,1]
        models[name] = {"model": model, "probabilities": prob}
        pred[f"{name}_prob_up"] = prob
    if "daily_text" in frame:
        tfidf = Pipeline([("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1,2), min_df=1, stop_words="english")),
                          ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))])
        tfidf.fit(train.daily_text.fillna(""), y_train)
        prob = tfidf.predict_proba(test.daily_text.fillna(""))[:,1]
        models["tfidf_text"] = {"model": tfidf, "probabilities": prob}
        pred["tfidf_text_prob_up"] = prob
    return y_test.to_numpy(), pred.reset_index(drop=True), models
