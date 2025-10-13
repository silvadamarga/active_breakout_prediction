#!/usr/bin/env python3
# train_breakout_single_task.py
"""
Time-series-safe single-task breakout prediction model.
- Per-ticker chronological splits (train/val/test).
- Scaler fit only on training sequences.
- Single-output NN: breakout_next_N_days (sigmoid).
- Sample weights to handle class imbalance.
"""

import os
import logging
import pickle
import random
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
import yfinance as yf
from sklearn.metrics import (accuracy_score, average_precision_score,
                           roc_auc_score)
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (Conv1D, Dense, Dropout, LSTM,
                                   MaxPooling1D)

# ----------------- Configuration -----------------
INITIAL_TICKERS = [
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM', 'V', 'JNJ',
    'UNH', 'HD', 'PG', 'MA', 'BAC', 'XOM', 'CVX', 'LLY', 'COST', 'PEP'
]
HISTORY_PERIOD = "5y"
TIME_STEPS = 60
BREAKOUT_PERIOD_DAYS = 15
BREAKOUT_THRESHOLD_PERCENT = 10.0
TEST_FRACTION = 0.20
VAL_FRACTION_OF_TEST = 0.5
MIN_HISTORY_REQUIREMENT = 252 * 2  # Require at least 2 years of data for robust features
MODEL_FILENAME = "breakout_model.keras" # Renamed model file
SCALER_FILENAME = "scaler.pkl"
METADATA_FILENAME = "metadata.pkl"
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 50

# ----------------- Logging -----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ----------------- Data Fetching Utilities -----------------
def get_most_active_stocks(limit: int = 50) -> List[str]:
    """Fetches 'most active' tickers from Yahoo; falls back to an empty list on failure."""
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
    """Returns (ticker, dataframe) or (ticker, None) on failure."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, auto_adjust=True)
        if df.empty:
            logging.warning(f"No data for {ticker}")
            return ticker, None
        return ticker, df.rename_axis('Date')
    except Exception as e:
        logging.error(f"Error fetching {ticker}: {e}")
        return ticker, None

def fetch_all_tickers(tickers: List[str], max_workers: int = 10) -> List[Tuple[str, pd.DataFrame]]:
    """Fetches tickers in parallel, returning a list of (ticker, df)."""
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = [f.result() for f in [ex.submit(get_stock_data, t) for t in tickers]]
    valid_results = [(t, df) for t, df in results if df is not None]
    logging.info(f"Fetched valid data for {len(valid_results)}/{len(tickers)} tickers.")
    return valid_results

# ----------------- Feature Engineering -----------------
def add_fundamentals(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds fundamental data. WARNING: This introduces lookahead bias as it fetches
    CURRENT fundamentals and applies them to historical data. Use with caution.
    """
    try:
        info = yf.Ticker(ticker).info
        df['PE_Ratio'] = info.get('trailingPE', np.nan)
        df['Dividend_Yield'] = info.get('dividendYield', 0.0) * 100
        # Forward-fill and then back-fill to handle NaNs
        return df.fillna(method='ffill').fillna(method='bfill')
    except Exception as e:
        logging.warning(f"Could not fetch fundamentals for {ticker}: {e}")
        df['PE_Ratio'] = np.nan
        df['Dividend_Yield'] = np.nan
        return df

def process_data_for_ticker(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    """Creates technical features and targets for a single ticker's dataframe."""
    if df is None or len(df) < MIN_HISTORY_REQUIREMENT:
        logging.warning(f"Skipping {ticker}: insufficient data ({len(df) if df is not None else 0} rows)")
        return None

    df = df.copy().sort_index()
    close = df['Close']
    
    # Technical Indicators
    df['SMA_50_rel'] = (close / close.rolling(50).mean()) - 1.0
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs.replace([np.inf, -np.inf], np.nan).fillna(0)))
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    df['MACD_rel'] = (exp12 - exp26) / close
    df['ROC_20'] = close.pct_change(20)
    df['Volume_rel'] = (df['Volume'] / df['Volume'].rolling(20).mean()) - 1.0
    tr = pd.concat([(df['High'] - df['Low']), (df['High'] - close.shift()).abs(), (df['Low'] - close.shift()).abs()], axis=1).max(axis=1)
    df['ATR_14_rel'] = tr.ewm(alpha=1/14, adjust=False).mean() / close
    sma_20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['BB_Width'] = (4 * std20) / sma_20
    obv = (np.sign(close.diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV_roc_20'] = obv.pct_change(20).replace([np.inf, -np.inf], np.nan)

    # Target: Breakout
    future_return = (close.shift(-BREAKOUT_PERIOD_DAYS) / close) - 1.0
    df['breakout_next_N'] = (future_return * 100.0 > BREAKOUT_THRESHOLD_PERCENT).astype(int)

    processed_df = df.dropna()
    logging.info(f"Processed {ticker}: {len(processed_df)} rows after feature engineering")
    return processed_df

# ----------------- Sequence Creation & Splits -----------------
def create_sequences_and_splits(
    ticker_dfs: List[Tuple[str, pd.DataFrame]],
    features: List[str],
    time_steps: int,
    test_fraction: float,
    val_fraction_of_test: float
):
    """Builds sequences for each ticker and splits them chronologically."""
    X_train, X_val, X_test = [], [], []
    y_bk_train, y_bk_val, y_bk_test = [], [], []

    for ticker, df in ticker_dfs:
        processed = process_data_for_ticker(ticker, df)
        if processed is None: continue

        data_vals = processed[features].values
        bk_labels = processed['breakout_next_N'].values
        
        n = len(processed)
        if n <= time_steps + 1:
            logging.warning(f"Skipping {ticker}: not enough data after processing ({n} rows)")
            continue

        reserved = int(n * test_fraction)
        train_end_idx = n - reserved
        val_end_idx = train_end_idx + int(reserved * val_fraction_of_test)

        for i in range(time_steps, n):
            seq = data_vals[i - time_steps:i]
            if i < train_end_idx:
                X_train.append(seq)
                y_bk_train.append(bk_labels[i])
            elif i < val_end_idx:
                X_val.append(seq)
                y_bk_val.append(bk_labels[i])
            else:
                X_test.append(seq)
                y_bk_test.append(bk_labels[i])
    
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)
    y_bk_train = np.array(y_bk_train)
    y_bk_val = np.array(y_bk_val)
    y_bk_test = np.array(y_bk_test)

    logging.info(f"Sequences created: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return (X_train, X_val, X_test, y_bk_train, y_bk_val, y_bk_test)

# ----------------- Model Architecture -----------------
def build_breakout_model(n_timesteps: int, n_features: int):
    """Builds the CNN-LSTM single-output model for breakout prediction."""
    inp = Input(shape=(n_timesteps, n_features), name="input_seq")
    x = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(inp)
    x = MaxPooling1D(pool_size=2)(x)
    
    # Using recurrent_dropout forces the standard, more compatible LSTM implementation
    x = LSTM(64, return_sequences=False, dropout=0.25, recurrent_dropout=0.25)(x)
    x = Dropout(0.3)(x)
    shared = Dense(64, activation='relu')(x)

    # Breakout Head
    bk_out = Dense(32, activation='relu', name='bk_head')(shared)
    bk_out = Dropout(0.2)(bk_out)
    bk_out = Dense(1, activation='sigmoid', name='breakout')(bk_out)

    # Model now has a single input and a single output
    model = Model(inputs=inp, outputs=bk_out)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy', # Simplified loss
        metrics=[
            'accuracy', 
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ] # Simplified metrics
    )
    return model

# ----------------- Main Pipeline -----------------
def main():
    """Orchestrates the entire ML pipeline for a single-task breakout model."""
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)
    logging.info("🚀 Initializing breakout prediction pipeline...")

    # Data Acquisition
    most_active = get_most_active_stocks(limit=30)
    tickers = sorted(list(set(INITIAL_TICKERS + most_active)))
    ticker_dfs = fetch_all_tickers(tickers, max_workers=10)
    if not ticker_dfs:
        logging.error("Fatal: No data could be fetched. Aborting.")
        return
    logging.info(f"✅ Fetched data for {len(ticker_dfs)} tickers.")

    # Feature Engineering & Sequencing
    features = [
        'SMA_50_rel', 'RSI_14', 'MACD_rel', 'ROC_20', 'Volume_rel',
        'ATR_14_rel', 'BB_Width', 'OBV_roc_20'
    ]
    # Note: The return signature for create_sequences_and_splits is now simpler
    (X_train, X_val, X_test,
     y_bk_train, y_bk_val, y_bk_test) = create_sequences_and_splits(
         ticker_dfs, features, TIME_STEPS, TEST_FRACTION, VAL_FRACTION_OF_TEST
     )
    if len(X_train) == 0:
        logging.error("Fatal: No training sequences created. Aborting.")
        return

    # Data Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    if len(X_val) > 0:
        X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    if len(X_test) > 0:
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # Sample Weights for breakout class imbalance
    unique_classes, counts = np.unique(y_bk_train, return_counts=True)
    logging.info(f"Breakout class distribution (train): {dict(zip(unique_classes, counts))}")
    
    breakout_sample_weights = np.ones_like(y_bk_train, dtype=float)
    if len(unique_classes) > 1:
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_bk_train)
        weight_map = dict(zip(unique_classes, class_weights))
        breakout_sample_weights = np.array([weight_map[label] for label in y_bk_train], dtype=float)

    # Model Build & Train
    model = build_breakout_model(TIME_STEPS, len(features)) # Call the new model function
    model.summary(print_fn=logging.info)

    callbacks = [
        # Monitor 'val_auc' now, aiming for the best classification threshold
        EarlyStopping(monitor='val_auc', mode='max', patience=8, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_FILENAME, monitor='val_auc', mode='max', save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)
    ]

    logging.info("🏋️ Starting model training...")
    model.fit(
        X_train, y_bk_train, # Simplified targets
        validation_data=(X_val, y_bk_val) if len(X_val) > 0 else None,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        sample_weight=breakout_sample_weights, # Simplified weights
        callbacks=callbacks,
        verbose=2
    )
    logging.info("✅ Training finished.")

    # Save Artifacts
    logging.info("💾 Saving model, scaler, and metadata...")
    metadata = {'features': features, 'used_tickers': [t for t, df in ticker_dfs]}
    with open(SCALER_FILENAME, 'wb') as f: pickle.dump(scaler, f)
    with open(METADATA_FILENAME, 'wb') as f: pickle.dump(metadata, f)

    # Evaluation
    if len(X_test) > 0:
        logging.info("📊 Evaluating model on the test set...")
        pred_bk = model.predict(X_test, batch_size=BATCH_SIZE * 2).ravel()

        acc_bk = accuracy_score(y_bk_test, (pred_bk > 0.5))
        auc_bk = roc_auc_score(y_bk_test, pred_bk) if len(np.unique(y_bk_test)) > 1 else 0.5
        ap_bk = average_precision_score(y_bk_test, pred_bk) if len(np.unique(y_bk_test)) > 1 else 0.5
        baseline_pr_auc = np.sum(y_bk_test) / len(y_bk_test) if len(y_bk_test) > 0 else 0

        logging.info("="*22 + " FINAL TEST METRICS " + "="*22)
        logging.info(f"BREAKOUT -> Accuracy: {acc_bk:.4f} | AUC: {auc_bk:.4f} | PR-AUC: {ap_bk:.4f} (Baseline: {baseline_pr_auc:.4f})")
        logging.info("="*64)
    else:
        logging.warning("No test set available for final evaluation.")

if __name__ == "__main__":
    main()