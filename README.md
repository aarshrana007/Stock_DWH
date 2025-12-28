# stock_dwh (drop-in directory)

This is a **from-scratch** mini data-warehouse + ML pipeline directory you can unzip directly into your repo.

## What it creates
- `warehouse/bronze` raw ingests (append-only, partitioned by `dt=YYYY-MM-DD`)
- `warehouse/silver` clean facts: `fact_news`, `fact_prices`
- `warehouse/gold` predictions + serving marts:
  - `fact_predictions`
  - `mart_market_snapshot_topbottom` (TOP 10 + BOTTOM 10)

## Inputs (you point to your existing folders/files)
Set env vars (locally, Kaggle, GitHub Actions, Streamlit secrets):
- `NEWS_GITHUB_CSV_PATH` (default: `data/news_data/historical_news.csv`)
- `NEWS_RSS_URLS` comma-separated (optional)
- `MARKET_CSV_PATH` (default: `data/market/ohlcv.csv`)

## Run locally
```bash
pip install -r requirements.txt

python -m stock_dwh.cli ingest
python -m stock_dwh.cli silver
python -m stock_dwh.cli infer
```

## Streamlit demo
```bash
streamlit run streamlit_app/app_stock_dwh_demo.py
```

## Notes
- Sentiment + model are **placeholders** (fast + safe). Replace later with FinBERT + LightGBM.
- News→stock mapping is optional. If your `fact_news` already has a `ticker` column, features will pick it up.
