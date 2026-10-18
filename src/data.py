import pandas as pd

def load_news(path):
    df = pd.read_csv(path)
    required = {"published_at", "ticker", "headline"}
    if not required.issubset(df.columns):
        raise ValueError(f"News CSV needs columns: {sorted(required)}")
    if "text" not in df:
        df["text"] = ""
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["headline"] = df["headline"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    return df.dropna(subset=["published_at"]).sort_values("published_at").reset_index(drop=True)

def load_prices(path):
    df = pd.read_csv(path)
    required = {"date", "ticker", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"Prices CSV needs columns: {sorted(required)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).query("close > 0").sort_values(["ticker", "date"])
    df["next_return"] = df.groupby("ticker")["close"].shift(-1) / df["close"] - 1
    df["target_up"] = (df["next_return"] > 0).astype("Int64")
    df.loc[df.groupby("ticker").tail(1).index, "target_up"] = pd.NA
    return df.reset_index(drop=True)
