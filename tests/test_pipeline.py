import pandas as pd
from src.sentiment import lexicon_scores
from src.features import aggregate_daily_news

def test_sentiment_polarity():
    x = lexicon_scores(["strong growth and profit", "warning of loss and decline", "ordinary statement"])
    assert x.loc[0,"sentiment_score"] > 0
    assert x.loc[1,"sentiment_score"] < 0
    assert x.loc[2,"sentiment_score"] == 0

def test_daily_aggregation():
    df = pd.DataFrame({"published_at":pd.to_datetime(["2025-01-01T10:00:00Z","2025-01-01T12:00:00Z"]),
      "ticker":["ABC","ABC"],"headline":["Strong growth","Profit rises"],
      "sentiment_score":[0.5,1.0],"sentiment_label":["positive","positive"]})
    result = aggregate_daily_news(df)
    assert len(result) == 1 and result.loc[0,"article_count"] == 2
