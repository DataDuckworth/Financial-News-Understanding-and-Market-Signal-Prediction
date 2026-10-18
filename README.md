# Financial News Understanding & Market Signal Prediction

Research project: sentiment extraction from financial news, daily feature aggregation, TF-IDF and structured ML baselines, and chronological out-of-sample evaluation.

**Educational use only.** Demo data is synthetic and does not demonstrate real predictive performance.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python -m src.run_pipeline --news data/sample_news.csv --prices data/sample_prices.csv --sentiment lexicon --output reports/demo
```
Optional FinBERT:
```bash
pip install transformers torch
python -m src.run_pipeline --news data/sample_news.csv --prices data/sample_prices.csv --sentiment finbert --output reports/finbert
```
Run tests: `pytest -q`

## Data schemas
`news.csv`: `published_at` (timestamp), `ticker`, `headline`, optional `text`.
`prices.csv`: `date`, `ticker`, `close`.

The target is the next observed trading session's return direction. The sample data is only a smoke test. For credible research, use a licensed real dataset, verify market-time cutoffs/time zones, use walk-forward evaluation, and account for costs/slippage. Never claim results you have not reproduced.

## Repository contents
- `src/sentiment.py`: lexicon baseline and optional FinBERT scoring
- `src/features.py`: daily news aggregation and market features
- `src/models.py`: TF-IDF, sentiment-only, market-only, combined models
- `src/evaluate.py`: classification metrics and illustrative long/cash backtest
- `src/run_pipeline.py`: command-line pipeline
- `tests/`: basic tests
