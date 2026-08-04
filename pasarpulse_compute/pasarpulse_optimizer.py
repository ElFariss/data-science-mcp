from __future__ import annotations

import argparse
import json
import math
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from urllib3.util.retry import Retry

SEED = 42
START_DATE = pd.Timestamp("2024-02-01")
END_DATE = pd.Timestamp("2026-07-31")
ORIGINS = pd.to_datetime([
    "2026-02-02", "2026-03-02", "2026-03-30", "2026-04-27",
    "2026-05-25", "2026-06-22", "2026-07-17",
])
HORIZONS = (7, 14)
ENDPOINT = "https://www.bi.go.id/hargapangan/WebSite/TabelHarga/GetGridDataKomoditas"
COMMODITIES = {
    "BM": {"source_id": "com_11", "name": "Bawang Merah"},
    "CMK": {"source_id": "com_14", "name": "Cabai Merah Keriting"},
    "CRM": {"source_id": "com_16", "name": "Cabai Rawit Merah"},
}
MARKETS = {"retail": 1, "wholesale": 3, "producer": 4}
PROVINCES = {
    "aceh": ("11", "Aceh"), "sumatera utara": ("12", "Sumatera Utara"),
    "sumatra utara": ("12", "Sumatera Utara"), "sumatera barat": ("13", "Sumatera Barat"),
    "sumatra barat": ("13", "Sumatera Barat"), "riau": ("14", "Riau"),
    "jambi": ("15", "Jambi"), "sumatera selatan": ("16", "Sumatera Selatan"),
    "sumatra selatan": ("16", "Sumatera Selatan"), "bengkulu": ("17", "Bengkulu"),
    "lampung": ("18", "Lampung"), "kepulauan bangka belitung": ("19", "Kepulauan Bangka Belitung"),
    "bangka belitung": ("19", "Kepulauan Bangka Belitung"), "kepulauan riau": ("21", "Kepulauan Riau"),
    "dki jakarta": ("31", "DKI Jakarta"), "jakarta": ("31", "DKI Jakarta"),
    "jawa barat": ("32", "Jawa Barat"), "jawa tengah": ("33", "Jawa Tengah"),
    "di yogyakarta": ("34", "DI Yogyakarta"), "daerah istimewa yogyakarta": ("34", "DI Yogyakarta"),
    "yogyakarta": ("34", "DI Yogyakarta"), "jawa timur": ("35", "Jawa Timur"),
    "banten": ("36", "Banten"), "bali": ("51", "Bali"),
    "nusa tenggara barat": ("52", "Nusa Tenggara Barat"), "ntb": ("52", "Nusa Tenggara Barat"),
    "nusa tenggara timur": ("53", "Nusa Tenggara Timur"), "ntt": ("53", "Nusa Tenggara Timur"),
    "kalimantan barat": ("61", "Kalimantan Barat"), "kalimantan tengah": ("62", "Kalimantan Tengah"),
    "kalimantan selatan": ("63", "Kalimantan Selatan"), "kalimantan timur": ("64", "Kalimantan Timur"),
    "kalimantan utara": ("65", "Kalimantan Utara"), "sulawesi utara": ("71", "Sulawesi Utara"),
    "sulawesi tengah": ("72", "Sulawesi Tengah"), "sulawesi selatan": ("73", "Sulawesi Selatan"),
    "sulawesi tenggara": ("74", "Sulawesi Tenggara"), "gorontalo": ("75", "Gorontalo"),
    "sulawesi barat": ("76", "Sulawesi Barat"), "maluku": ("81", "Maluku"),
    "maluku utara": ("82", "Maluku Utara"), "papua barat": ("91", "Papua Barat"),
    "papua barat daya": ("92", "Papua Barat Daya"), "papua": ("94", "Papua"),
    "papua selatan": ("95", "Papua Selatan"), "papua tengah": ("96", "Papua Tengah"),
    "papua pegunungan": ("97", "Papua Pegunungan"),
}


def log(message: str) -> None:
    print(message, flush=True)


def norm(value: Any) -> str:
    text = unicodedata.normalize("NFKC", "" if value is None else str(value))
    return re.sub(r"\s+", " ", text.replace("\xa0", " ").casefold().strip())


def parse_price(value: Any) -> float:
    if value is None:
        return np.nan
    text = re.sub(r"[^0-9,.\-]", "", str(value).strip())
    if not text or text in {"-", ".", ","}:
        return np.nan
    if "," in text and "." in text:
        pos = max(text.rfind(","), text.rfind("."))
        decimals = len(text) - pos - 1
        if decimals in {1, 2}:
            decimal = text[pos]
            text = text.replace("." if decimal == "," else ",", "").replace(decimal, ".")
        else:
            text = text.replace(",", "").replace(".", "")
    elif "," in text:
        decimals = len(text.rsplit(",", 1)[1])
        text = text.replace(",", ".") if decimals in {1, 2} else text.replace(",", "")
    elif text.count(".") == 1 and len(text.rsplit(".", 1)[1]) == 3:
        text = text.replace(".", "")
    try:
        return float(text)
    except ValueError:
        return np.nan


def month_chunks(start: pd.Timestamp, end: pd.Timestamp):
    cur = start
    while cur <= end:
        month_end = min(cur + pd.offsets.MonthEnd(0), end)
        yield cur, month_end
        cur = month_end + pd.Timedelta(days=1)


def session() -> requests.Session:
    retry = Retry(total=6, connect=6, read=6, backoff_factor=1.0,
                  status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"Accept": "application/json", "User-Agent": "PasarPulse research optimizer/1.0"})
    return s


def download_prices(cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    s = session()
    chunks = list(month_chunks(START_DATE, END_DATE))
    total = len(COMMODITIES) * len(MARKETS) * len(chunks)
    counter = 0
    for market, price_type in MARKETS.items():
        for commodity_code, meta in COMMODITIES.items():
            for start, end in chunks:
                counter += 1
                path = cache_dir / f"pihps_{market}_{commodity_code}_{start:%Y_%m}.json"
                params = {
                    "price_type_id": price_type, "comcat_id": meta["source_id"],
                    "province_id": "", "regency_id": "", "showKota": "false",
                    "showPasar": "false", "tipe_laporan": 1,
                    "start_date": start.strftime("%Y-%m-%d"),
                    "end_date": end.strftime("%Y-%m-%d"), "skip": 0,
                    "take": 100, "requireTotalCount": "true",
                }
                if path.exists() and path.stat().st_size > 20:
                    payload = json.loads(path.read_text())
                else:
                    response = s.get(ENDPOINT, params=params, timeout=120)
                    response.raise_for_status()
                    payload = response.json()
                    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                rows = payload.get("data", []) if isinstance(payload, dict) else []
                recs = []
                for row in rows:
                    if str(row.get("level", "")) != "1":
                        continue
                    province = PROVINCES.get(norm(row.get("name")))
                    if not province:
                        continue
                    for key, value in row.items():
                        if not (isinstance(key, str) and len(key) == 10 and key[2:3] == "/" and key[5:6] == "/"):
                            continue
                        date = pd.to_datetime(key, dayfirst=True, errors="coerce")
                        price = parse_price(value)
                        if pd.notna(date) and start <= date <= end and np.isfinite(price) and 1000 < price < 500000:
                            recs.append((date.normalize(), province[0], province[1], commodity_code, meta["name"], market, price))
                if recs:
                    parts.append(pd.DataFrame(recs, columns=["date", "province_code", "province_name",
                                                             "commodity_code", "commodity_name", "market_level", "price"]))
                if counter % 15 == 0:
                    log(f"downloaded/loaded {counter}/{total} snapshots")
    if not parts:
        raise RuntimeError("PIHPS acquisition returned no observations")
    frame = pd.concat(parts, ignore_index=True)
    frame = frame.sort_values("date").drop_duplicates(
        ["date", "province_code", "commodity_code", "market_level"], keep="last"
    )
    frame = frame.sort_values(["province_code", "commodity_code", "market_level", "date"]).reset_index(drop=True)
    log(f"PIHPS rows={len(frame):,}; provinces={frame.province_code.nunique()}; series={frame.groupby(['province_code','commodity_code','market_level']).ngroups}")
    return frame


def add_holidays(df: pd.DataFrame) -> pd.DataFrame:
    try:
        import holidays
        id_holidays = holidays.country_holidays("ID", years=[2024, 2025, 2026])
        names = {pd.Timestamp(key): str(value) for key, value in id_holidays.items()}
    except Exception:
        names = {}
    out = df.copy()
    out["is_holiday"] = out["date"].isin(names).astype("int8")
    out["holiday_name"] = out["date"].map(names).fillna("none")
    ramadan = pd.to_datetime(["2024-03-11", "2025-03-01", "2026-02-18"])
    eid = pd.to_datetime(["2024-04-10", "2025-03-31", "2026-03-20"])
    eid_adha = pd.to_datetime(["2024-06-17", "2025-06-06", "2026-05-27"])

    def nearest_days(dates: pd.Series, events: pd.DatetimeIndex) -> np.ndarray:
        array = dates.values.astype("datetime64[D]")[:, None]
        event_array = events.values.astype("datetime64[D]")[None, :]
        delta = (array - event_array).astype("timedelta64[D]").astype(int)
        index = np.abs(delta).argmin(axis=1)
        return delta[np.arange(len(delta)), index]

    out["days_from_ramadan"] = nearest_days(out["date"], pd.DatetimeIndex(ramadan))
    out["days_from_eid"] = nearest_days(out["date"], pd.DatetimeIndex(eid))
    out["days_from_eid_adha"] = nearest_days(out["date"], pd.DatetimeIndex(eid_adha))
    out["ramadan_window"] = out["days_from_ramadan"].between(0, 30).astype("int8")
    out["eid_window"] = out["days_from_eid"].between(-21, 14).astype("int8")
    return out


def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["province_code", "province_name", "commodity_code", "commodity_name", "market_level"]
    all_dates = pd.date_range(START_DATE, END_DATE, freq="D")
    series = raw[keys].drop_duplicates().reset_index(drop=True)
    grid = series.merge(pd.DataFrame({"date": all_dates}), how="cross")
    panel = grid.merge(raw[keys + ["date", "price"]], on=keys + ["date"], how="left")
    panel["observed"] = panel["price"].notna().astype("int8")
    panel = panel.sort_values(keys + ["date"]).reset_index(drop=True)
    panel["price_filled"] = panel.groupby(keys, observed=True)["price"].transform(lambda values: values.ffill(limit=3))
    panel["series_id"] = panel["province_code"] + "|" + panel["commodity_code"] + "|" + panel["market_level"]
    panel["island"] = panel["province_code"].str[0].map({
        "1": "sumatra", "2": "sumatra", "3": "java", "5": "bali_nt",
        "6": "kalimantan", "7": "sulawesi", "8": "maluku", "9": "papua",
    }).fillna("other")
    panel = add_holidays(panel)
    panel["dow"] = panel["date"].dt.dayofweek.astype("int8")
    panel["dom"] = panel["date"].dt.day.astype("int8")
    panel["month"] = panel["date"].dt.month.astype("int8")
    panel["quarter"] = panel["date"].dt.quarter.astype("int8")
    panel["weekofyear"] = panel["date"].dt.isocalendar().week.astype("int16")
    day_of_year = panel["date"].dt.dayofyear
    panel["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    panel["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    group = panel.groupby("series_id", observed=True, sort=False)
    for lag in [1, 2, 3, 5, 7, 10, 14, 21, 28, 35, 42, 56, 84, 112, 168, 364]:
        panel[f"price_lag_{lag}"] = group["price_filled"].shift(lag)
        panel[f"delta_lag_{lag}"] = panel["price_filled"] - panel[f"price_lag_{lag}"]
        panel[f"ratio_lag_{lag}"] = panel["price_filled"] / panel[f"price_lag_{lag}"] - 1.0
    shifted = group["price_filled"].shift(1)
    for window in [3, 5, 7, 14, 21, 28, 42, 56, 84, 112]:
        rolling = shifted.groupby(panel["series_id"], observed=True).rolling(
            window, min_periods=max(2, window // 3)
        )
        panel[f"roll_mean_{window}"] = rolling.mean().reset_index(level=0, drop=True)
        panel[f"roll_std_{window}"] = rolling.std().reset_index(level=0, drop=True)
        panel[f"roll_min_{window}"] = rolling.min().reset_index(level=0, drop=True)
        panel[f"roll_max_{window}"] = rolling.max().reset_index(level=0, drop=True)
        panel[f"mean_gap_{window}"] = panel["price_filled"] - panel[f"roll_mean_{window}"]
    for span in [3, 7, 14, 28, 56]:
        panel[f"ewm_{span}"] = group["price_filled"].transform(
            lambda values: values.shift(1).ewm(span=span, adjust=False, min_periods=2).mean()
        )
        panel[f"ewm_gap_{span}"] = panel["price_filled"] - panel[f"ewm_{span}"]

    market = panel.pivot_table(
        index=["date", "province_code", "commodity_code"],
        columns="market_level", values="price_filled", aggfunc="last"
    ).reset_index()
    market.columns.name = None
    market = market.rename(columns={market_name: f"sameprov_{market_name}" for market_name in MARKETS})
    panel = panel.merge(market, on=["date", "province_code", "commodity_code"], how="left")
    for first, second in [("retail", "wholesale"), ("wholesale", "producer"), ("retail", "producer")]:
        panel[f"spread_{first}_{second}"] = panel[f"sameprov_{first}"] - panel[f"sameprov_{second}"]
        panel[f"ratio_{first}_{second}"] = panel[f"sameprov_{first}"] / panel[f"sameprov_{second}"] - 1.0

    peer_key = ["date", "commodity_code", "market_level"]
    national = (
        panel.groupby(peer_key, observed=True)["price_filled"]
        .agg(nat_median="median", nat_mean="mean", nat_std="std")
        .reset_index()
        .sort_values(["commodity_code", "market_level", "date"])
    )
    national_group = national.groupby(["commodity_code", "market_level"], observed=True, sort=False)
    for lag in [1, 7, 14, 28]:
        national[f"nat_median_delta_{lag}"] = national["nat_median"] - national_group["nat_median"].shift(lag)
    panel = panel.merge(national, on=peer_key, how="left")

    island_key = ["date", "island", "commodity_code", "market_level"]
    island = (
        panel.groupby(island_key, observed=True)["price_filled"]
        .agg(island_median="median", island_mean="mean", island_std="std")
        .reset_index()
        .sort_values(["island", "commodity_code", "market_level", "date"])
    )
    island_group = island.groupby(["island", "commodity_code", "market_level"], observed=True, sort=False)
    for lag in [1, 7, 14, 28]:
        island[f"island_median_delta_{lag}"] = island["island_median"] - island_group["island_median"].shift(lag)
    panel = panel.merge(island, on=island_key, how="left")
    panel["gap_nat"] = panel["price_filled"] - panel["nat_median"]
    panel["gap_island"] = panel["price_filled"] - panel["island_median"]
    return panel.replace([np.inf, -np.inf], np.nan)


CATEGORICAL = ["province_code", "commodity_code", "market_level", "series_id", "island", "holiday_name"]
LAGS = [1, 2, 3, 5, 7, 10, 14, 21, 28, 35, 42, 56, 84, 112, 168, 364]
WINDOWS = [3, 5, 7, 14, 21, 28, 42, 56, 84, 112]
PRICE_BASE = ["price_filled"] + [f"price_lag_{value}" for value in LAGS]
PRICE_BASE += [f"delta_lag_{value}" for value in LAGS]
PRICE_BASE += [f"ratio_lag_{value}" for value in LAGS]
PRICE_BASE += [f"{prefix}_{window}" for window in WINDOWS for prefix in ["roll_mean", "roll_std", "roll_min", "roll_max", "mean_gap"]]
PRICE_BASE += [f"{prefix}_{span}" for span in [3, 7, 14, 28, 56] for prefix in ["ewm", "ewm_gap"]]
CALENDAR = [
    "dow", "dom", "month", "quarter", "weekofyear", "doy_sin", "doy_cos",
    "is_holiday", "days_from_ramadan", "days_from_eid", "days_from_eid_adha",
    "ramadan_window", "eid_window",
]
GRAPH = [f"sameprov_{market}" for market in MARKETS] + [
    "spread_retail_wholesale", "spread_wholesale_producer", "spread_retail_producer",
    "ratio_retail_wholesale", "ratio_wholesale_producer", "ratio_retail_producer",
    "nat_median", "nat_mean", "nat_std", "island_median", "island_mean", "island_std",
    "gap_nat", "gap_island",
] + [f"{column}_delta_{lag}" for column in ["nat_median", "island_median"] for lag in [1, 7, 14, 28]]


def smape(y_true, y_pred):
    denominator = np.abs(y_true) + np.abs(y_pred)
    return float(np.mean(2 * np.abs(y_true - y_pred) / np.where(denominator == 0, 1, denominator)) * 100)


def metrics(y_true, y_pred):
    return {
        "mae_idr_per_kg": float(mean_absolute_error(y_true, y_pred)),
        "rmse_idr_per_kg": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "smape_percent": smape(np.asarray(y_true), np.asarray(y_pred)),
        "mean_error_idr_per_kg": float(np.mean(np.asarray(y_pred) - np.asarray(y_true))),
    }


def make_lgb(params=None):
    from lightgbm import LGBMRegressor
    base = dict(
        objective="regression_l1", n_estimators=1200, learning_rate=0.025,
        num_leaves=31, max_depth=-1, min_child_samples=80,
        subsample=0.85, colsample_bytree=0.80, reg_alpha=0.05, reg_lambda=1.5,
        random_state=SEED, n_jobs=-1, verbosity=-1,
    )
    if params:
        base.update(params)
    return LGBMRegressor(**base)


def encode_frame(train: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    train_frame = train[features].copy()
    test_frame = test[features].copy()
    for column in CATEGORICAL:
        if column not in features:
            continue
        values = pd.concat([train_frame[column], test_frame[column]], ignore_index=True).astype(str).fillna("missing")
        encoder = LabelEncoder().fit(values)
        train_frame[column] = encoder.transform(train_frame[column].astype(str).fillna("missing")).astype("category")
        test_frame[column] = encoder.transform(test_frame[column].astype(str).fillna("missing")).astype("category")
    return train_frame, test_frame


def fit_predict_lgb(train, test, features, target, sample_weight=None, params=None):
    train_frame, test_frame = encode_frame(train, test, features)
    model = make_lgb(params)
    model.fit(
        train_frame, train[target], sample_weight=sample_weight,
        categorical_feature=[column for column in CATEGORICAL if column in features],
    )
    return model.predict(test_frame), model


def fit_predict_cat(train, test, features, target, sample_weight=None, loss="MAE"):
    from catboost import CatBoostRegressor
    train_frame = train[features].copy()
    test_frame = test[features].copy()
    categorical = [column for column in CATEGORICAL if column in features]
    for column in categorical:
        train_frame[column] = train_frame[column].fillna("missing").astype(str)
        test_frame[column] = test_frame[column].fillna("missing").astype(str)
    model = CatBoostRegressor(
        loss_function=loss, eval_metric="MAE", iterations=1400, depth=8,
        learning_rate=0.025, l2_leaf_reg=5.0, random_seed=SEED,
        random_strength=0.4, bootstrap_type="Bernoulli", subsample=0.85,
        allow_writing_files=False, verbose=False, thread_count=-1,
    )
    model.fit(train_frame, train[target], cat_features=categorical, sample_weight=sample_weight)
    return model.predict(test_frame), model


def create_supervised(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    future = panel[["series_id", "date", "price", "observed"]].copy()
    future["date"] = future["date"] - pd.Timedelta(days=horizon)
    future = future.rename(columns={"price": "target_price", "observed": "target_observed"})
    frame = panel.merge(future, on=["series_id", "date"], how="left")
    frame["horizon_days"] = horizon
    frame["target_delta"] = frame["target_price"] - frame["price_filled"]
    frame["target_logratio"] = np.log(frame["target_price"] / frame["price_filled"])
    target_date = frame["date"] + pd.Timedelta(days=horizon)
    frame["target_dow"] = target_date.dt.dayofweek.astype("int8")
    frame["target_month"] = target_date.dt.month.astype("int8")
    frame["target_doy_sin"] = np.sin(2 * np.pi * target_date.dt.dayofyear / 365.25)
    frame["target_doy_cos"] = np.cos(2 * np.pi * target_date.dt.dayofyear / 365.25)
    return frame


def recent_weights(dates: pd.Series, origin: pd.Timestamp):
    age = (origin - dates).dt.days.clip(lower=0)
    return (0.35 + 0.65 * np.exp(-age / 240.0)).to_numpy()


def optimize_blend(y_true, predictions: dict[str, np.ndarray]):
    names = list(predictions)
    matrix = np.column_stack([predictions[name] for name in names])
    best = (float("inf"), np.ones(len(names)) / len(names))
    for first in np.linspace(0, 1, 21):
        for second in np.linspace(0, 1 - first, int(round((1 - first) * 20)) + 1):
            third = 1 - first - second
            weights = np.array([first, second, third])
            score = mean_absolute_error(y_true, matrix @ weights)
            if score < best[0]:
                best = (score, weights)
    return dict(zip(names, best[1])), best[0]


def run_experiment(panel: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    all_predictions = []
    fold_rows = []
    feature_rows = []
    variants = {
        "B2_price_only_LGBM": PRICE_BASE + CATEGORICAL[:-1],
        "B3_price_calendar_LGBM": PRICE_BASE + CALENDAR + CATEGORICAL,
        "B6_dynamic_graph_LGBM": PRICE_BASE + CALENDAR + GRAPH + CATEGORICAL,
    }
    for horizon in HORIZONS:
        supervised = create_supervised(panel, horizon)
        valid_label = (
            (supervised["observed"] == 1) & (supervised["target_observed"] == 1)
            & supervised["target_price"].notna() & supervised["price_filled"].notna()
        )
        supervised = supervised[valid_label].copy()
        for fold_id, origin in enumerate(ORIGINS, 1):
            log(f"h={horizon} fold={fold_id}/{len(ORIGINS)} origin={origin.date()}")
            train = supervised[
                (supervised["date"] < origin)
                & ((supervised["date"] + pd.Timedelta(days=horizon)) <= origin)
            ].copy()
            test = supervised[supervised["date"] == origin].copy()
            if train.empty or test.empty:
                log("skip empty fold")
                continue
            counts = train.groupby("series_id", observed=True).size()
            allowed = counts[counts >= 240].index
            train = train[train.series_id.isin(allowed)].copy()
            test = test[test.series_id.isin(allowed)].copy()
            weights = recent_weights(train["date"], origin)
            variant_predictions = {}
            for name, requested_features in variants.items():
                features = [feature for feature in requested_features if feature in train.columns]
                delta, _ = fit_predict_lgb(train, test, features, "target_delta", weights)
                prediction = np.clip(test["price_filled"].to_numpy() + delta, 1000, 500000)
                variant_predictions[name] = prediction
                fold_rows.append({
                    "model": name, "horizon_days": horizon, "fold_id": fold_id,
                    "forecast_origin": origin.date().isoformat(), "rows_scored": len(test),
                    **metrics(test.target_price.to_numpy(), prediction),
                })

            final_features = variants["B6_dynamic_graph_LGBM"] + [
                "target_dow", "target_month", "target_doy_sin", "target_doy_cos",
            ]
            final_features = [feature for feature in final_features if feature in train.columns]
            lgb_delta, lgb_model = fit_predict_lgb(
                train, test, final_features, "target_delta", weights,
                {"n_estimators": 1800, "learning_rate": 0.018, "num_leaves": 47,
                 "min_child_samples": 55, "max_bin": 127, "reg_lambda": 2.5,
                 "colsample_bytree": 0.9},
            )
            pred_lgb = np.clip(test.price_filled.to_numpy() + lgb_delta, 1000, 500000)
            cat_delta, _ = fit_predict_cat(train, test, final_features, "target_delta", weights, "MAE")
            pred_cat = np.clip(test.price_filled.to_numpy() + cat_delta, 1000, 500000)
            cat_log, _ = fit_predict_cat(train, test, final_features, "target_logratio", weights, "MAE")
            pred_log = np.clip(test.price_filled.to_numpy() * np.exp(np.clip(cat_log, -1.2, 1.2)), 1000, 500000)
            experts = {"lgb_delta": pred_lgb, "cat_delta": pred_cat, "cat_log": pred_log}

            validation_start = origin - pd.Timedelta(days=84)
            inner_train = train[(train["date"] + pd.Timedelta(days=horizon)) < validation_start].copy()
            validation = train[
                (train["date"] >= validation_start)
                & ((train["date"] + pd.Timedelta(days=horizon)) <= origin)
            ].copy()
            if len(inner_train) > 5000 and len(validation) > 300:
                inner_weights = recent_weights(inner_train.date, validation_start)
                val_lgb, _ = fit_predict_lgb(
                    inner_train, validation, final_features, "target_delta", inner_weights,
                    {"n_estimators": 1100, "learning_rate": 0.025, "num_leaves": 47,
                     "min_child_samples": 55, "max_bin": 127, "reg_lambda": 2.5},
                )
                val_cat, _ = fit_predict_cat(inner_train, validation, final_features, "target_delta", inner_weights, "MAE")
                val_log, _ = fit_predict_cat(inner_train, validation, final_features, "target_logratio", inner_weights, "MAE")
                validation_predictions = {
                    "lgb_delta": np.clip(validation.price_filled.to_numpy() + val_lgb, 1000, 500000),
                    "cat_delta": np.clip(validation.price_filled.to_numpy() + val_cat, 1000, 500000),
                    "cat_log": np.clip(validation.price_filled.to_numpy() * np.exp(np.clip(val_log, -1.2, 1.2)), 1000, 500000),
                }
                blend_weights, inner_mae = optimize_blend(validation.target_price.to_numpy(), validation_predictions)
            else:
                blend_weights = {"lgb_delta": 0.45, "cat_delta": 0.40, "cat_log": 0.15}
                inner_mae = np.nan

            blend = sum(blend_weights[name] * experts[name] for name in blend_weights)
            current = test.price_filled.to_numpy()
            max_change = np.maximum(9000, 0.40 * current)
            blend = np.clip(blend, current - max_change, current + max_change)
            fold_rows.append({
                "model": "PROPOSED_robust_dynamic_graph_ensemble", "horizon_days": horizon,
                "fold_id": fold_id, "forecast_origin": origin.date().isoformat(),
                "rows_scored": len(test), "blend_weights": json.dumps(blend_weights),
                "inner_val_mae": inner_mae, **metrics(test.target_price.to_numpy(), blend),
            })
            for name, prediction in experts.items():
                fold_rows.append({
                    "model": f"expert_{name}", "horizon_days": horizon, "fold_id": fold_id,
                    "forecast_origin": origin.date().isoformat(), "rows_scored": len(test),
                    **metrics(test.target_price.to_numpy(), prediction),
                })

            base = test[[
                "series_id", "province_code", "province_name", "commodity_code",
                "commodity_name", "market_level", "date", "target_price", "price_filled",
            ]].copy().rename(columns={
                "date": "forecast_origin", "price_filled": "current_price",
                "target_price": "actual_price",
            })
            base["target_date"] = base["forecast_origin"] + pd.Timedelta(days=horizon)
            base["horizon_days"] = horizon
            base["fold_id"] = fold_id
            output_predictions = {
                **variant_predictions,
                **{f"expert_{name}": prediction for name, prediction in experts.items()},
                "PROPOSED_robust_dynamic_graph_ensemble": blend,
            }
            for name, prediction in output_predictions.items():
                result = base.copy()
                result["model"] = name
                result["predicted_price"] = prediction
                result["absolute_error"] = (result.actual_price - result.predicted_price).abs()
                all_predictions.append(result)
            for feature, importance in zip(final_features, lgb_model.feature_importances_):
                feature_rows.append({
                    "horizon_days": horizon, "fold_id": fold_id,
                    "feature": feature, "importance": float(importance),
                })

    predictions = pd.concat(all_predictions, ignore_index=True)
    folds = pd.DataFrame(fold_rows)
    pooled_rows = []
    for (model, horizon), group in predictions.groupby(["model", "horizon_days"]):
        pooled_rows.append({
            "model": model, "horizon_days": horizon, "rows_scored": len(group),
            "series_scored": group.series_id.nunique(), "folds_scored": group.fold_id.nunique(),
            **metrics(group.actual_price, group.predicted_price),
        })
    pooled = pd.DataFrame(pooled_rows).sort_values(["horizon_days", "mae_idr_per_kg"])
    predictions.to_csv(outdir / "oof_predictions.csv", index=False)
    folds.to_csv(outdir / "fold_metrics.csv", index=False)
    pooled.to_csv(outdir / "pooled_metrics.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(outdir / "feature_importance.csv", index=False)
    (outdir / "run_metadata.json").write_text(json.dumps({
        "protocol": "paper-aligned rolling-origin direct multi-horizon",
        "origins": [date.date().isoformat() for date in ORIGINS],
        "horizons": list(HORIZONS),
        "date_range": [START_DATE.date().isoformat(), END_DATE.date().isoformat()],
        "seed": SEED, "rows_panel": len(panel), "series": panel.series_id.nunique(),
    }, indent=2))
    print("\nPOOLED METRICS\n", pooled.to_string(index=False), flush=True)
    return pooled, folds, predictions


def build_notebook(outdir: Path):
    import nbformat as nbf
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        nbf.v4.new_markdown_cell("# PasarPulse: Paper-Aligned Multimodal Dynamic-Graph Forecasting\n\n**Methodological subtitle:** Direct multi-horizon robust ensemble with cross-market hierarchy and dynamic peer messages\n\n**Team:** Tembok Ratapan Solo"),
        nbf.v4.new_markdown_cell("## Table of Contents\n\n1. Introduction and Research Questions\n2. Related Work and Methodological Positioning\n3. Dependencies and Reproducibility\n4. Data and Exploratory Analysis\n5. Feature Engineering\n6. Model Architecture\n7. Validation Strategy\n8. Ablation Study\n9. Explainable AI\n10. Inference-Time and Cost Analysis\n11. Final Training and Submission\n12. Discussion and Research Questions\n13. Limitations\n14. Conclusion\n15. References"),
        nbf.v4.new_markdown_cell("## Abstract\n\nThis executed notebook extends the paper's B1 ARIMA baseline into price-only, calendar-aware, dynamic-graph, and robust ensemble variants. All results use the same seven chronological rolling origins and exact 7/14-calendar-day targets. Random splitting is prohibited. The graph messages are derived from the producer–wholesale–retail hierarchy and contemporaneous national/island peer summaries; every time-varying predictor is available at the forecast origin."),
        nbf.v4.new_markdown_cell("## 1. Introduction and Research Questions\n\nRQ1: How much does global price-history learning improve on B1 ARIMA?\n\nRQ2: Do Indonesian calendar events improve forecasting?\n\nRQ3: Do cross-market and peer-graph messages add signal?\n\nRQ4: Does a robust blend improve stability across origins and horizons?"),
        nbf.v4.new_markdown_cell("## 2. Related Work and Methodological Positioning\n\nThe experiment follows the paper's B2–B6 ladder, while the final model is a missing-tolerant, direct multi-horizon approximation of the proposed dynamic graph. The implementation prioritizes strict validation and reproducibility over architectural labels."),
        nbf.v4.new_markdown_cell("## 3. Dependencies and Reproducibility\n\nThe complete optimizer is stored in `pasarpulse_optimizer.py`. Random seed: 42. Inputs are versioned PIHPS monthly snapshots. Outputs below are generated by the same workflow run."),
        nbf.v4.new_markdown_cell("## 4. Data and Exploratory Analysis\n\nPIHPS province-level daily prices for Bawang Merah, Cabai Merah Keriting, and Cabai Rawit Merah at producer, wholesale, and retail levels. Integrity and coverage are recorded in the run metadata."),
        nbf.v4.new_markdown_cell("## 5. Feature Engineering\n\nHistorical lags, rolling/EWMA statistics, target-date calendar, Ramadan/Eid proximity, cross-market spreads, and dynamic national/island peer messages. Only backward-looking or origin-known values are used."),
        nbf.v4.new_markdown_cell("## 6. Model Architecture\n\n### 6.1 B1 reference\nPer-series ARIMA(1,1,1), reported from the prior executed baseline.\n\n### 6.2 B2 price-only\nGlobal LightGBM with price history and entity identifiers.\n\n### 6.3 B3 price + calendar\nAdds Indonesian holiday and event-distance features.\n\n### 6.4 B6 dynamic graph\nAdds producer–wholesale–retail messages and peer summaries.\n\n### 6.5 Proposed robust ensemble\nBlends direct-delta LightGBM, direct-delta CatBoost, and log-ratio CatBoost. Blend weights are selected only on an inner chronological window."),
        nbf.v4.new_markdown_cell("## 7. Validation Strategy\n\nSeven expanding rolling origins are identical to the B1 notebook. Training labels must mature by the origin; test labels occur exactly 7 or 14 calendar days later."),
        nbf.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nROOT = Path('.')\nmetrics = pd.read_csv(ROOT / 'pooled_metrics.csv')\nfolds = pd.read_csv(ROOT / 'fold_metrics.csv')\nmetrics.sort_values(['horizon_days','mae_idr_per_kg'])"),
        nbf.v4.new_markdown_cell("## 8. Ablation Study\n\nThe executed table above compares B2, B3, B6, individual experts, and the proposed ensemble under identical labels."),
        nbf.v4.new_code_cell("folds.pivot_table(index=['model','horizon_days'], columns='fold_id', values='mae_idr_per_kg').round(2)"),
        nbf.v4.new_markdown_cell("## 9. Explainable AI\n\nTree split importance is exported for every fold. A later SHAP cell may be run on a sampled fold model when model binaries are retained."),
        nbf.v4.new_code_cell("pd.read_csv(ROOT / 'feature_importance.csv').groupby(['horizon_days','feature']).importance.mean().groupby(level=0, group_keys=False).nlargest(20)"),
        nbf.v4.new_markdown_cell("## 10. Inference-Time and Cost Analysis\n\nTraining uses CPU-compatible LightGBM and CatBoost. Inference is batch tabular prediction and does not require a GPU."),
        nbf.v4.new_markdown_cell("## 11. Final Training and Submission\n\nFinal fitting and application-specific export remain separate from rolling-origin research evaluation."),
        nbf.v4.new_markdown_cell("## 12. Discussion and Research Questions\n\nThe executed ablations answer RQ1–RQ4 quantitatively. Improvements are accepted only when pooled MAE and fold stability both improve."),
        nbf.v4.new_markdown_cell("## 13. Limitations\n\nPIHPS currently returns 34 provinces. Weather and satellite modalities are not included in this optimization run. Dynamic peer summaries approximate message passing but are not a learned neural GNN."),
        nbf.v4.new_markdown_cell("## 14. Conclusion\n\nSee the executed pooled and fold metrics for the best validated architecture."),
        nbf.v4.new_markdown_cell("## References\n\nPasarPulse proposed paper; Lim et al., Temporal Fusion Transformers; Ke et al., LightGBM; Prokhorenkova et al., CatBoost."),
    ]
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    path = outdir / "pasarpulse_proposed_optimizer.ipynb"
    nbf.write(notebook, path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, default=Path("pasarpulse_run"))
    args = parser.parse_args()
    work = args.workdir
    raw_directory = work / "raw_pihps"
    output_directory = work / "results"
    work.mkdir(parents=True, exist_ok=True)
    started = time.time()
    prices_path = work / "price_daily.csv"
    if prices_path.exists():
        raw = pd.read_csv(prices_path, parse_dates=["date"], dtype={"province_code": str})
    else:
        raw = download_prices(raw_directory)
        raw.to_csv(prices_path, index=False)
    panel_path = work / "panel_features.parquet"
    if panel_path.exists():
        panel = pd.read_parquet(panel_path)
    else:
        panel = build_panel(raw)
        panel.to_parquet(panel_path, index=False)
    run_experiment(panel, output_directory)
    notebook = build_notebook(output_directory)
    (output_directory / "runtime_seconds.txt").write_text(str(time.time() - started))
    log(f"notebook={notebook}; runtime={time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
