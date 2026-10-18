import argparse
from pathlib import Path
import pandas as pd
from src.data import load_news, load_prices
from src.sentiment import score_news
from src.features import aggregate_daily_news, make_model_frame
from src.models import fit_compare_models
from src.evaluate import classification_metrics, simple_directional_backtest, save_metrics

def run(news_path, prices_path, sentiment_mode, output_dir):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    news, prices = load_news(news_path), load_prices(prices_path)
    scored = score_news(news, sentiment_mode)
    daily = aggregate_daily_news(scored)
    text = scored.copy()
    text["date"] = text.published_at.dt.tz_convert(None).dt.normalize()
    text["article"] = (text.headline.fillna("") + ". " + text.text.fillna("")).str.strip()
    text = text.groupby(["ticker","date"], as_index=False).agg(daily_text=("article", lambda x: " ".join(x)))
    frame = make_model_frame(prices, daily).merge(text, on=["ticker","date"], how="left")
    frame["daily_text"] = frame.daily_text.fillna("")
    y, predictions, models = fit_compare_models(frame)
    metrics = {}
    for name, obj in models.items():
        col = f"{name}_prob_up"
        metrics[name] = {"classification": classification_metrics(y,obj["probabilities"]),
                         "backtest": simple_directional_backtest(predictions,col)}
    scored.to_csv(out/"scored_news.csv", index=False)
    daily.to_csv(out/"daily_features.csv", index=False)
    predictions.to_csv(out/"predictions.csv", index=False)
    save_metrics(metrics, out/"metrics.json")
    print("Saved reports to", out.resolve())
    print(pd.DataFrame({k:v["classification"] for k,v in metrics.items()}).T)

def main():
    parser = argparse.ArgumentParser(description="Financial news sentiment and market signal research pipeline")
    parser.add_argument("--news", required=True)
    parser.add_argument("--prices", required=True)
    parser.add_argument("--sentiment", choices=["lexicon","finbert"], default="lexicon")
    parser.add_argument("--output", default="reports/run")
    a = parser.parse_args()
    run(a.news,a.prices,a.sentiment,a.output)

if __name__ == "__main__":
    main()
