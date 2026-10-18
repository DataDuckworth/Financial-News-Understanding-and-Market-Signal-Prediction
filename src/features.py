def aggregate_daily_news(scored):
    import pandas as pd
    df = scored.copy()
    df["date"] = df["published_at"].dt.tz_convert(None).dt.normalize()
    return df.groupby(["ticker", "date"], as_index=False).agg(
        sentiment_mean=("sentiment_score", "mean"),
        sentiment_std=("sentiment_score", "std"),
        positive_share=("sentiment_label", lambda x: (x == "positive").mean()),
        negative_share=("sentiment_label", lambda x: (x == "negative").mean()),
        article_count=("headline", "count"))

def make_model_frame(prices, daily):
    import pandas as pd
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    grp = frame.groupby("ticker")["close"]
    frame["return_1d"] = grp.pct_change()
    frame["return_5d"] = grp.pct_change(5)
    frame["volatility_5d"] = frame.groupby("ticker")["return_1d"].rolling(5, min_periods=2).std().reset_index(level=0, drop=True)
    frame = frame.merge(daily, on=["ticker", "date"], how="left")
    cols = ["sentiment_mean", "sentiment_std", "positive_share", "negative_share", "article_count"]
    frame[cols] = frame[cols].fillna(0).fillna(0)
    return frame.dropna(subset=["target_up"]).assign(target_up=lambda x: x["target_up"].astype(int)).sort_values(["date","ticker"]).reset_index(drop=True)
