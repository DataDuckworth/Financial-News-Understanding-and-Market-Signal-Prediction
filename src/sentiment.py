import re
import pandas as pd

POS = set("beat beats growth grow profit profitable surge strong upgrade upgraded gain gains record bullish outperform positive rally raises raised improve improved".split())
NEG = set("miss misses loss losses decline declines weak downgrade downgraded fall falls drop drops bearish negative cuts cut lawsuit fraud warning warns slump lower".split())

def lexicon_scores(texts):
    rows = []
    for text in texts:
        tokens = re.findall(r"[a-z]+", str(text).lower())
        p, n = sum(t in POS for t in tokens), sum(t in NEG for t in tokens)
        score = (p - n) / max(p + n, 1)
        rows.append({"sentiment_score": score,
                     "sentiment_label": "positive" if score > 0 else "negative" if score < 0 else "neutral",
                     "sentiment_positive": p / max(p+n, 1), "sentiment_negative": n / max(p+n, 1)})
    return pd.DataFrame(rows)

def finbert_scores(texts, model_name="ProsusAI/finbert", batch_size=16):
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError("Install optional dependencies with: pip install transformers torch") from exc
    clf = pipeline("text-classification", model=model_name, tokenizer=model_name,
                   return_all_scores=True, truncation=True)
    texts = [str(t)[:4000] or " " for t in texts]
    raw = []
    for i in range(0, len(texts), batch_size):
        raw.extend(clf(texts[i:i+batch_size], batch_size=batch_size))
    rows = []
    for scores in raw:
        probs = {x["label"].lower(): float(x["score"]) for x in scores}
        p, n = probs.get("positive", 0), probs.get("negative", 0)
        rows.append({"sentiment_score": p-n, "sentiment_label": max(probs, key=probs.get),
                     "sentiment_positive": p, "sentiment_negative": n})
    return pd.DataFrame(rows)

def score_news(news, mode="lexicon"):
    texts = (news["headline"].fillna("") + ". " + news["text"].fillna("")).tolist()
    if mode == "lexicon":
        scores = lexicon_scores(texts)
    elif mode == "finbert":
        scores = finbert_scores(texts)
    else:
        raise ValueError("mode must be lexicon or finbert")
    scores.index = news.index
    return pd.concat([news.copy(), scores], axis=1)
