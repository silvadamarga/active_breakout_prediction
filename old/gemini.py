#!/usr/bin/env python3
# train_breakout_multi_task.py
"""
Time-series-safe multi-task breakout + direction model.
- Per-ticker chronological splits (train/val/test)
- Scaler fit only on training sequences
- Multi-output NN: direction_next_day (sigmoid), breakout_next_N_days (sigmoid)
- Sample weights to handle breakout class imbalance in training
"""

import os
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ----------------- Configuration -----------------
INITIAL_TICKERS = [
    'AAPL','MSFT','NVDA',
#    'AVGO','ADBE','CRM','ORCL','SNPS','CDNS',
#    'LLY','JNJ','UNH','MRK','ABBV','PFE','TMO',
#    'BRK-B','JPM','V','MA','BAC','AXP',
#    'AMZN','TSLA','HD','MCD','NKE','SBUX',
#    'GOOGL','META','VZ','CMCSA','NFLX','TMUS',
#    'CAT','UNP','BA','UPS','DAL',
#    'PG','COST','WMT','KO','MDLZ',
#    'XOM','CVX','COP','SLB',
#    'NEE','DUK','SO',
#    'PLD','AMT','SPG',
    'LIN','SHW','FCX'
]
HISTORY_PERIOD = "5y"               # yfinance period string
TIME_STEPS = 60                     # sequence length in days
BREAKOUT_PERIOD_DAYS = 15           # horizon for breakout target
BREAKOUT_THRESHOLD_PERCENT = 10.0   # breakout defined as > this percent
TEST_FRACTION = 0.20                # fraction per-ticker reserved as test+val
VAL_FRACTION_OF_TEST = 0.5          # split the reserved fraction into val/test (50/50)
MIN_HISTORY_REQUIREMENT = 252      # require at least this many rows for processing
MODEL_FILENAME = "breakout_multitask.keras"
SCALER_FILENAME = "scaler.pkl"
METADATA_FILENAME = "metadata.pkl"
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 50

# ----------------- Logging -----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ----------------- Utilities -----------------
def get_most_active_stocks(limit: int = 50) -> List[str]:
    """Attempt to fetch 'most active' tickers from Yahoo; fallback to empty list if fails."""
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=most_actives"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        quotes = data.get("finance", {}).get("result", [{}])[0].get("quotes", [])
        tickers = [q["symbol"] for q in quotes if "symbol" in q]
        logging.info(f"Fetched {len(tickers)} most-active tickers from Yahoo.")
        return tickers[:limit]
    except Exception as e:
        logging.warning(f"Could not fetch most active tickers: {e}")
        return []

def get_stock_data(ticker: str, period: str = HISTORY_PERIOD) -> Tuple[str, pd.DataFrame]:
    """Return (ticker, dataframe) or (ticker, None) if failure."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, auto_adjust=True)
        if df.empty:
            logging.warning(f"No data for {ticker}")
            return ticker, None
        df = df.rename_axis('Date')
        return ticker, df
    except Exception as e:
        logging.error(f"Error fetching {ticker}: {e}")
        return ticker, None

def fetch_all_tickers(tickers: List[str], max_workers: int = 8) -> List[Tuple[str, pd.DataFrame]]:
    """Fetch tickers in parallel; returns list of (ticker, df)."""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(get_stock_data, t) for t in tickers]
        for f in futures:
            t, df = f.result()
            results.append((t, df))
    return results

# ----------------- Feature Engineering -----------------
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    rs = rs.replace([np.inf, -np.inf], np.nan).fillna(0)
    return 100 - 100 / (1 + rs)

def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def process_data_for_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """Create technical features and targets. Returns processed df with a 'direction' and 'breakout' column."""
    if df is None or len(df) < MIN_HISTORY_REQUIREMENT:
        return None

    df = df.copy()
    df = df.sort_index()

    close = df['Close']
    
    # --- IMPROVEMENT 1: Make price-based features stationary (relative to current price) ---
    # This helps the model generalize better by not focusing on absolute price levels.
    df['SMA_50_rel'] = (close / close.rolling(50).mean()) - 1.0
    df['RSI_14'] = compute_rsi(close, period=14)
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    df['MACD_rel'] = (exp12 - exp26) / close
    df['ROC_20'] = close.pct_change(20) * 100.0
    
    volume_sma_20 = df['Volume'].rolling(20).mean()
    df['Volume_rel'] = (df['Volume'] / volume_sma_20) - 1.0
    df['Volume_Spike'] = (df['Volume'] > volume_sma_20 * 1.5).astype(int)
    
    df['ATR_14_rel'] = compute_atr(df['High'], df['Low'], close, period=14) / close
    
    rolling_max_52w = close.rolling(window=252).max()
    df['Dist_from_52W_High'] = (close - rolling_max_52w) / rolling_max_52w * 100.0

    # Bollinger Bands
    sma_20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['BB_Upper_rel'] = ((sma_20 + 2 * std20) / close) - 1.0
    df['BB_Lower_rel'] = ((sma_20 - 2 * std20) / close) - 1.0
    df['BB_Width'] = (df['BB_Upper_rel'] - df['BB_Lower_rel'])

    # OBV (On-Balance Volume) - already relative, so no change needed
    df['OBV'] = (np.sign(close.diff()) * df['Volume']).fillna(0).cumsum()

    # Targets:
    df['return_1d_next'] = close.shift(-1) / close - 1.0
    df['direction_next_day'] = (df['return_1d_next'] > 0).astype(int)

    future_return = close.shift(-BREAKOUT_PERIOD_DAYS) / close - 1.0
    df['breakout_next_N'] = (future_return.abs() * 100.0 > BREAKOUT_THRESHOLD_PERCENT).astype(int)

    df = df.dropna()
    return df

# ----------------- Sequence creation & splits -----------------
def create_sequences_and_splits(
    ticker_dfs: List[Tuple[str, pd.DataFrame]],
    features: List[str],
    time_steps: int = TIME_STEPS,
    test_fraction: float = TEST_FRACTION,
    val_fraction_of_test: float = VAL_FRACTION_OF_TEST
):
    """
    For each ticker build sequences and assign them chronologically into train/val/test.
    """
    X_train_s, X_val_s, X_test_s = [], [], []
    y_dir_train_s, y_dir_val_s, y_dir_test_s = [], [], []
    y_bk_train_s, y_bk_val_s, y_bk_test_s = [], [], []

    for ticker, df in ticker_dfs:
        processed = process_data_for_ticker(df)
        if processed is None or processed.empty:
            continue

        n = len(processed)
        if n <= time_steps + 1:
            continue

        reserved = int(n * test_fraction)
        if reserved < 2:
            continue
        train_end_idx = n - reserved
        val_size = int(reserved * val_fraction_of_test)
        val_end_idx = train_end_idx + val_size
        
        data_vals = processed[features].values
        dir_labels = processed['direction_next_day'].values
        bk_labels = processed['breakout_next_N'].values

        for i in range(time_steps, n):
            seq = data_vals[i - time_steps:i]
            dir_lab = dir_labels[i]
            bk_lab = bk_labels[i]
            if i < train_end_idx:
                X_train_s.append(seq); y_dir_train_s.append(dir_lab); y_bk_train_s.append(bk_lab)
            elif i < val_end_idx:
                X_val_s.append(seq); y_dir_val_s.append(dir_lab); y_bk_val_s.append(bk_lab)
            else:
                X_test_s.append(seq); y_dir_test_s.append(dir_lab); y_bk_test_s.append(bk_lab)

    X_train = np.array(X_train_s); X_val = np.array(X_val_s); X_test = np.array(X_test_s)
    y_dir_train = np.array(y_dir_train_s); y_dir_val = np.array(y_dir_val_s); y_dir_test = np.array(y_dir_test_s)
    y_bk_train = np.array(y_bk_train_s); y_bk_val = np.array(y_bk_val_s); y_bk_test = np.array(y_bk_test_s)

    logging.info(f"Sequences created: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    logging.info(f"Train shapes: X={X_train.shape}, y_dir={y_dir_train.shape}, y_bk={y_bk_train.shape}")
    
    return (X_train, X_val, X_test,
            y_dir_train, y_dir_val, y_dir_test,
            y_bk_train, y_bk_val, y_bk_test)

# ----------------- Model -----------------
def build_multitask_model(n_timesteps: int, n_features: int, l2_reg: float = 0.001):
    inp = Input(shape=(n_timesteps, n_features), name="input_seq")
    x = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(inp)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.25)(x)
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.3)(x)
    shared = Dense(64, activation='relu')(x)

    dir_out = Dense(32, activation='relu')(shared)
    dir_out = Dropout(0.2)(dir_out)
    dir_out = Dense(1, activation='sigmoid', name='direction')(dir_out)

    bk_out = Dense(32, activation='relu')(shared)
    bk_out = Dropout(0.2)(bk_out)
    bk_out = Dense(1, activation='sigmoid', name='breakout')(bk_out)

    model = Model(inputs=inp, outputs=[dir_out, bk_out])
    model.compile(optimizer='adam',
                  loss={'direction': 'binary_crossentropy', 'breakout': 'binary_crossentropy'},
                  metrics={'direction': ['accuracy', tf.keras.metrics.AUC(name='auc')],
                           'breakout': ['accuracy', tf.keras.metrics.AUC(name='auc')]})
    return model

# ----------------- Main training flow -----------------
def main():
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    most_active = get_most_active_stocks(limit=30)
    tickers = sorted(list(set(INITIAL_TICKERS + most_active)))
    logging.info(f"Total tickers to fetch: {len(tickers)}")

    fetched = fetch_all_tickers(tickers, max_workers=10)
    ticker_dfs = [(t, df) for (t, df) in fetched if df is not None]
    used_tickers = [t for t, df in ticker_dfs]
    logging.info(f"Fetched data for {len(ticker_dfs)} tickers (non-empty).")

    if not ticker_dfs:
        logging.error("No data fetched, aborting.")
        return

    # --- UPDATED: New feature list reflecting the stationary changes ---
    features = [
        'SMA_50_rel', 'RSI_14', 'MACD_rel', 'ROC_20', 'Volume_rel', 'Volume_Spike',
        'ATR_14_rel', 'Dist_from_52W_High', 'BB_Upper_rel', 'BB_Lower_rel', 'BB_Width', 'OBV'
    ]

    (X_train, X_val, X_test,
     y_dir_train, y_dir_val, y_dir_test,
     y_bk_train, y_bk_val, y_bk_test) = create_sequences_and_splits(
         ticker_dfs, features, TIME_STEPS, TEST_FRACTION, VAL_FRACTION_OF_TEST)

    if len(X_train) == 0:
        logging.error("No training sequences — adjust MIN_HISTORY_REQUIREMENT / tickers.")
        return

    # Scaling
    ns, nt, nf = X_train.shape
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_flat = X_train.reshape((-1, nf))
    scaler.fit(X_train_flat)
    X_train = scaler.transform(X_train_flat).reshape((ns, nt, nf))

    if len(X_val) > 0:
        X_val = scaler.transform(X_val.reshape((-1, nf))).reshape((X_val.shape[0], nt, nf))
    if len(X_test) > 0:
        X_test = scaler.transform(X_test.reshape((-1, nf))).reshape((X_test.shape[0], nt, nf))

    # Sample weights
    if len(y_bk_train) > 0:
        classes = np.unique(y_bk_train)
        cw = compute_class_weight('balanced', classes=classes, y=y_bk_train)
        class_weight_map = {int(c): float(w) for c, w in zip(classes, cw)}
        sample_weights = np.array([class_weight_map[int(lbl)] for lbl in y_bk_train])
        logging.info(f"Breakout class weights (train): {class_weight_map}")
    else:
        sample_weights = None

    # Build / Train model
    model = build_multitask_model(TIME_STEPS, len(features))
    model.summary(print_fn=logging.info)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_FILENAME, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    y_train_targets = {'direction': y_dir_train, 'breakout': y_bk_train}
    y_val_targets = {'direction': y_dir_val, 'breakout': y_bk_val}

    logging.info("Starting training...")
    history = model.fit(X_train, y_train_targets,
                        validation_data=(X_val, y_val_targets) if len(X_val) > 0 else None,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        sample_weight=sample_weights,
                        callbacks=callbacks,
                        verbose=2)

    # Save scaler & metadata
    with open(SCALER_FILENAME, 'wb') as f:
        pickle.dump(scaler, f)
        
    # --- IMPROVEMENT 2: Add used tickers to metadata for reproducibility ---
    metadata = {
        'features': features,
        'used_tickers': used_tickers,
        'time_steps': TIME_STEPS,
        'breakout_period_days': BREAKOUT_PERIOD_DAYS,
        'breakout_threshold_percent': BREAKOUT_THRESHOLD_PERCENT
    }
    with open(METADATA_FILENAME, 'wb') as f:
        pickle.dump(metadata, f)
    logging.info(f"Saved scaler to {SCALER_FILENAME} and metadata to {METADATA_FILENAME}")

    # Evaluation on TEST set
    if len(X_test) == 0:
        logging.warning("No test set available to evaluate.")
        return

    preds = model.predict(X_test, batch_size=128)
    pred_dir, pred_bk = preds[0], preds[1]
    pred_dir, pred_bk = pred_dir.ravel(), pred_bk.ravel()

    auc_dir = roc_auc_score(y_dir_test, pred_dir) if len(np.unique(y_dir_test)) > 1 else np.nan
    auc_bk = roc_auc_score(y_bk_test, pred_bk) if len(np.unique(y_bk_test)) > 1 else np.nan
    ap_bk = average_precision_score(y_bk_test, pred_bk) if len(np.unique(y_bk_test)) > 1 else np.nan
    acc_dir = accuracy_score(y_dir_test, (pred_dir > 0.5).astype(int))
    acc_bk = accuracy_score(y_bk_test, (pred_bk > 0.5).astype(int))

    logging.info("=== TEST METRICS ===")
    logging.info(f"Direction: accuracy={acc_dir:.4f}, AUC={auc_dir:.4f}")
    logging.info(f"Breakout:  accuracy={acc_bk:.4f}, AUC={auc_bk:.4f}, PR-AUC={ap_bk:.4f}")
    
    logging.info("Training complete. Model & artifacts saved.")

if __name__ == "__main__":
    main()