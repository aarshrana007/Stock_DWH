print("✅ CLI LOADED:", __file__)
def infer_run() -> None:
    paths = get_paths()
    print("WAREHOUSE PATH:", paths.warehouse)
    print("BRONZE PATH:", paths.bronze)
    log = get_logger("stock_dwh.infer", paths.logs / "infer.log")

    # ---------------- Load silver data ----------------
    news_root = wh_join(paths.silver, "fact_news")
    px_root = wh_join(paths.silver, "fact_prices")

    news_files = list_parquet_files(news_root)
    px_files = list_parquet_files(px_root)

    fact_news = (
        pd.concat([read_parquet(p) for p in news_files], ignore_index=True)
        if news_files else pd.DataFrame()
    )
    fact_px = (
        pd.concat([read_parquet(p) for p in px_files], ignore_index=True)
        if px_files else pd.DataFrame()
    )

    # ---------------- Load NIFTY-50 universe ----------------
    universe_path = "Data/market/nifty50.csv"
    universe = pd.read_csv(universe_path)
    universe["ticker"] = universe["ticker"].str.upper()

    if fact_px.empty:
        log.warning("No prices found in silver. Universe will still be emitted.")
        fact_px = universe.copy()
        fact_px["has_price"] = False
    else:
        fact_px["ticker"] = fact_px["ticker"].str.upper()

        # LEFT JOIN → keep all 50 stocks (OLD BEHAVIOR)
        fact_px = universe.merge(
            fact_px,
            on="ticker",
            how="left"
        )

        fact_px["has_price"] = fact_px["close"].notna()

    # ---------------- Sentiment ----------------
    scored_news = (
        finbert_placeholder(fact_news)
        if not fact_news.empty else pd.DataFrame()
    )

    # ---------------- Feature build (ONLY where price exists) ----------------
    fact_px_valid = fact_px[fact_px["has_price"]].copy()

    if fact_px_valid.empty:
        log.warning("No stocks with valid prices. Skipping model inference.")
        preds = universe.copy()
        preds["pred"] = None
        preds["signal_status"] = "NO_PRICE"
    else:
        fact_px_valid["ts_utc"] = pd.to_datetime(fact_px_valid["ts_utc"], utc=True)
        asof = fact_px_valid["ts_utc"].max()

        feats = build_features(fact_px_valid, scored_news, asof)

        model_path = paths.artifacts / "models" / "champion" / "model.pkl"
        model = load_model(model_path)
        preds_valid = predict(model, feats)

        preds_valid["signal_status"] = "MODEL"

        # bring back full universe
        preds = universe.merge(
            preds_valid,
            on="ticker",
            how="left"
        )

        preds.loc[preds["pred"].isna(), "signal_status"] = "NO_PRICE"

    preds["dt"] = pd.Timestamp.utcnow().date().astype(str)

    # ---------------- Gold ----------------
    _write_partitioned(preds, wh_join(paths.gold, "fact_predictions"))

    # top / bottom snapshot
    ranked = preds[preds["pred"].notna()].sort_values("pred", ascending=False)
    top = ranked.head(10).assign(bucket="TOP")
    bot = ranked.tail(10).assign(bucket="BOTTOM")
    mart = pd.concat([top, bot], ignore_index=True)

    _write_partitioned(
        mart,
        wh_join(paths.gold, "mart_market_snapshot_topbottom"),
        filename="snapshot.parquet",
    )

    log.info("Inference done with full NIFTY-50 universe.")
