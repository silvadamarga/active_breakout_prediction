#!/usr/bin/env python3
# train_breakout_tuning.py
"""
Hyperparameter-tuned, time-series-safe, multi-task breakout prediction model.

This script uses KerasTuner to systematically find the optimal architecture and
training parameters for the multi-task model.

Key Enhancements:
- KerasTuner Integration: Automates the search for the best combination of
  Conv1D filters, LSTM/Dense units, dropout rates, and learning rate.
- Hyperband Algorithm: Efficiently searches the hyperparameter space to find a
  top-performing model configuration.
- Two-Phase Execution: First, it runs the tuning search, then it retrains the
  best model found on the full dataset before final evaluation.
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
import pandas_ta as ta
import keras_tuner as kt
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             roc_auc_score, average_precision_score)
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Input, Model, optimizers
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.layers import (Conv1D, Dense, Dropout, LSTM,
                                     MaxPooling1D)

# ----------------- Configuration -----------------
INITIAL_TICKERS = [
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM', 'V', 'JNJ',
    'UNH', 'HD', 'PG', 'MA', 'BAC', 'XOM', 'CVX', 'LLY', 'COST', 'PEP'
]
HISTORY_PERIOD = "5y"
TIME_STEPS = 60
BREAKOUT_PERIOD_DAYS = 10
BREAKOUT_THRESHOLD_PERCENT = 5.0
TEST_FRACTION = 0.20
VAL_FRACTION_OF_TEST = 0.5
MIN_HISTORY_REQUIREMENT = 252 * 2
MODEL_FILENAME = "breakout_tuned_model.keras"
SCALER_FILENAME = "scaler.pkl"
METADATA_FILENAME = "metadata.pkl"
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 50
# KerasTuner specific configuration
TUNER_EPOCHS = 15 # Epochs to run for each trial
MAX_TRIALS = 20   # Number of hyperparameter combinations to test
EXECUTION_PER_TRIAL = 2 # Number of models to train for each trial

# ----------------- Logging -----------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# ----------------- Data Fetching -----------------
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

# ----------------- Feature Engineering (with Manual BBands) -----------------
def process_data_for_ticker(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    """Creates enhanced technical features and multi-task targets."""
    if df is None or len(df) < MIN_HISTORY_REQUIREMENT:
        return None

    df = df.copy().sort_index()
    close = df['Close']
    
    df['SMA_50_rel'] = (close / close.rolling(50).mean()) - 1.0
    df['ROC_20'] = close.pct_change(20)
    df['Volume_rel'] = (df['Volume'] / df['Volume'].rolling(20).mean()) - 1.0
    
    # --- DEFINITIVE MANUAL BOLLINGER BAND CALCULATION ---
    bb_length = 20
    bb_std = 2.0
    rolling_mean = df['Close'].rolling(window=bb_length).mean()
    rolling_std = df['Close'].rolling(window=bb_length).std()
    
    df['BBM'] = rolling_mean
    df['BBU'] = rolling_mean + (rolling_std * bb_std)
    df['BBL'] = rolling_mean - (rolling_std * bb_std)
    df['BB_Width'] = (df['BBU'] - df['BBL']) / df['BBM']

    # --- Other Enhanced Features ---
    df.ta.obv(append=True)
    df['OBV_roc_20'] = df['OBV'].pct_change(20).replace([np.inf, -np.inf], np.nan)
    df.ta.rsi(length=14, append=True)
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    if isinstance(macd, pd.DataFrame) and 'MACD_12_26_9' in macd.columns:
        df['MACD_rel'] = macd['MACD_12_26_9'] / close
    else:
        df['MACD_rel'] = np.nan
        
    df.ta.adx(length=14, append=True)
    atr = df.ta.atr(length=14)
    df['ATR_14_rel'] = atr / close
    
    # --- Multi-Task Targets ---
    future_return = (close.shift(-BREAKOUT_PERIOD_DAYS) / close) - 1.0
    df['future_return_N'] = future_return * 100.0
    df['breakout_next_N'] = (df['future_return_N'] > BREAKOUT_THRESHOLD_PERCENT).astype(int)

    # --- Final Cleanup ---
    df.drop(columns=['BBM', 'BBU', 'BBL'], inplace=True, errors='ignore')
    return df.dropna()

# ----------------- Sequence Creation -----------------
def create_sequences_and_splits(
    ticker_dfs: List[Tuple[str, pd.DataFrame]],
    features: List[str],
    time_steps: int,
    test_fraction: float,
    val_fraction_of_test: float
):
    X_train, X_val, X_test = [], [], []
    y_bk_train, y_bk_val, y_bk_test = [], [], []
    y_ret_train, y_ret_val, y_ret_test = [], [], []

    for ticker, df in ticker_dfs:
        processed = process_data_for_ticker(ticker, df)
        if processed is None: continue
        data_vals = processed[features].values
        bk_labels = processed['breakout_next_N'].values
        ret_labels = processed['future_return_N'].values.clip(-50, 100)
        n = len(processed)
        if n <= time_steps + 1: continue
        reserved = int(n * test_fraction)
        train_end_idx = n - reserved
        val_end_idx = train_end_idx + int(reserved * val_fraction_of_test)
        for i in range(time_steps, n):
            seq = data_vals[i - time_steps:i]
            if i < train_end_idx:
                X_train.append(seq)
                y_bk_train.append(bk_labels[i])
                y_ret_train.append(ret_labels[i])
            elif i < val_end_idx:
                X_val.append(seq)
                y_bk_val.append(bk_labels[i])
                y_ret_val.append(ret_labels[i])
            else:
                X_test.append(seq)
                y_bk_test.append(bk_labels[i])
                y_ret_test.append(ret_labels[i])
    return (np.array(X_train), np.array(X_val), np.array(X_test),
            np.array(y_bk_train), np.array(y_bk_val), np.array(y_bk_test),
            np.array(y_ret_train), np.array(y_ret_val), np.array(y_ret_test))

# ----------------- Model Builder for KerasTuner -----------------
def build_model_for_tuner(hp):
    """Builds a tunable multi-task model."""
    n_timesteps = TIME_STEPS
    features = [
        'SMA_50_rel', 'ROC_20', 'Volume_rel', 'BB_Width', 'OBV_roc_20',
        'RSI_14', 'MACD_rel', 'ADX_14', 'ATR_14_rel'
    ]
    n_features = len(features)

    inp = Input(shape=(n_timesteps, n_features), name="input_seq")
    
    hp_conv_filters = hp.Int('conv_filters', min_value=32, max_value=96, step=32)
    hp_lstm_units = hp.Int('lstm_units', min_value=32, max_value=96, step=32)
    hp_shared_units = hp.Int('shared_units', min_value=32, max_value=96, step=32)
    hp_head_units = hp.Int('head_units', min_value=16, max_value=48, step=16)
    hp_dropout_rate = hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])

    x = Conv1D(filters=hp_conv_filters, kernel_size=5, activation='relu', padding='same')(inp)
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(units=hp_lstm_units, return_sequences=False, dropout=hp_dropout_rate, recurrent_dropout=hp_dropout_rate)(x)
    x = Dropout(hp_dropout_rate)(x)
    shared_trunk = Dense(units=hp_shared_units, activation='relu', name='shared_trunk')(x)

    bk_head = Dense(units=hp_head_units, activation='relu')(shared_trunk)
    bk_head = Dropout(hp_dropout_rate)(bk_head)
    breakout_output = Dense(1, activation='sigmoid', name='breakout')(bk_head)

    ret_head = Dense(units=hp_head_units, activation='relu')(shared_trunk)
    ret_head = Dropout(hp_dropout_rate)(ret_head)
    return_output = Dense(1, activation='linear', name='return')(ret_head)

    model = Model(inputs=inp, outputs=[breakout_output, return_output])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=hp_learning_rate),
        loss={'breakout': 'binary_crossentropy', 'return': 'mean_squared_error'},
        loss_weights={'breakout': 1.2, 'return': 1.0},
        metrics={
            'breakout': [tf.keras.metrics.AUC(name='auc')],
            'return': ['mae']
        }
    )
    return model

# ----------------- Main Pipeline -----------------
def main():
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)
    logging.info("🚀 Initializing HYPERPARAMETER TUNING pipeline...")

    # --- 1. Data Preparation ---
    most_active = get_most_active_stocks(limit=30)
    tickers = sorted(list(set(INITIAL_TICKERS + most_active)))
    ticker_dfs = fetch_all_tickers(tickers, max_workers=10)
    if not ticker_dfs:
        logging.error("Fatal: No data could be fetched. Aborting.")
        return
    
    features = [
        'SMA_50_rel', 'ROC_20', 'Volume_rel', 'BB_Width', 'OBV_roc_20',
        'RSI_14', 'MACD_rel', 'ADX_14', 'ATR_14_rel'
    ]
    (X_train, X_val, X_test,
     y_bk_train, y_bk_val, y_bk_test,
     y_ret_train, y_ret_val, y_ret_test) = create_sequences_and_splits(
        ticker_dfs, features, TIME_STEPS, TEST_FRACTION, VAL_FRACTION_OF_TEST
     )
    if len(X_train) == 0:
        logging.error("Fatal: No training sequences created. Aborting.")
        return

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    
    unique, _ = np.unique(y_bk_train, return_counts=True)
    if len(unique) > 1:
        class_weights = compute_class_weight('balanced', classes=unique, y=y_bk_train)
        weight_map = dict(zip(unique, class_weights))
        breakout_sample_weights = np.array([weight_map[label] for label in y_bk_train])
    else:
        breakout_sample_weights = np.ones_like(y_bk_train, dtype=float)

    # --- FIX APPLIED HERE ---
    # Create lists for labels and sample weights to ensure their structures match.
    # The order must match the model's output order: [breakout, return].
    y_train_list = [y_bk_train, y_ret_train]
    y_val_list = [y_bk_val, y_ret_val]
    sample_weight_list = [
        breakout_sample_weights,
        np.ones_like(y_ret_train, dtype=float)
    ]

    # --- 2. Hyperparameter Search ---
    tuner = kt.Hyperband(
        build_model_for_tuner,
        objective=kt.Objective("val_breakout_auc", direction="max"),
        max_epochs=TUNER_EPOCHS,
        factor=3,
        directory='keras_tuner_dir',
        project_name='breakout_tuning',
        overwrite=True
    )
    
    stop_early = EarlyStopping(monitor='val_loss', patience=5)
    
    logging.info("⚙️ Starting hyperparameter search...")
    tuner.search(
        X_train_scaled,
        y_train_list, # Use list of labels
        validation_data=(X_val_scaled, y_val_list), # Use list of validation labels
        epochs=TUNER_EPOCHS,
        batch_size=BATCH_SIZE,
        sample_weight=sample_weight_list, # Use list of weights
        callbacks=[stop_early]
    )

    tuner.results_summary()
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    logging.info(f"🏆 Best Hyperparameters Found: \n {best_hps.values}")

    # --- 3. Retrain Best Model ---
    logging.info("🏋️ Retraining the best model on the full training data...")
    best_model = tuner.get_best_models(num_models=1)[0]

    X_full_train = np.concatenate([X_train_scaled, X_val_scaled], axis=0)
    y_bk_full_train = np.concatenate([y_bk_train, y_bk_val], axis=0)
    y_ret_full_train = np.concatenate([y_ret_train, y_ret_val], axis=0)
    
    # --- FIX APPLIED HERE ---
    # Create lists for the full training data to match structures.
    y_full_train_list = [y_bk_full_train, y_ret_full_train]

    unique_full, _ = np.unique(y_bk_full_train, return_counts=True)
    if len(unique_full) > 1:
        class_weights_full = compute_class_weight('balanced', classes=unique_full, y=y_bk_full_train)
        weight_map_full = dict(zip(unique_full, class_weights_full))
        full_sample_weights = np.array([weight_map_full[label] for label in y_bk_full_train])
    else:
        full_sample_weights = np.ones_like(y_bk_full_train, dtype=float)

    full_weights_list = [
        full_sample_weights,
        np.ones_like(y_ret_full_train, dtype=float)
    ]

    final_callbacks = [
        EarlyStopping(monitor='loss', patience=5, verbose=1, restore_best_weights=True)
    ]

    best_model.fit(
        X_full_train,
        y_full_train_list, # Use list of labels
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        sample_weight=full_weights_list, # Use list of weights
        callbacks=final_callbacks,
        verbose=2
    )
    logging.info("✅ Best model retrained.")

    # --- 4. Final Evaluation ---
    if len(X_test) > 0:
        X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        logging.info("📊 Evaluating tuned model on the test set...")
        pred_bk, pred_ret = best_model.predict(X_test_scaled, batch_size=BATCH_SIZE * 2)
        pred_bk, pred_ret = pred_bk.ravel(), pred_ret.ravel()
        
        auc_bk = roc_auc_score(y_bk_test, pred_bk) if len(np.unique(y_bk_test)) > 1 else 0.5
        ap_bk = average_precision_score(y_bk_test, pred_bk) if len(np.unique(y_bk_test)) > 1 else 0.5
        mae_ret = mean_absolute_error(y_ret_test, pred_ret)
        rmse_ret = np.sqrt(mean_squared_error(y_ret_test, pred_ret))

        logging.info("="*22 + " FINAL TUNED METRICS " + "="*22)
        logging.info(f"BREAKOUT -> AUC: {auc_bk:.4f} | PR-AUC: {ap_bk:.4f}")
        logging.info(f"RETURN   -> MAE: {mae_ret:.2f}% | RMSE: {rmse_ret:.2f}%")
        logging.info("="*65)
    
    # --- 5. Save Artifacts ---
    logging.info("💾 Saving tuned model, scaler, and metadata...")
    best_model.save(MODEL_FILENAME)
    metadata = {'features': features, 'used_tickers': [t for t, df in ticker_dfs]}
    with open(SCALER_FILENAME, 'wb') as f: pickle.dump(scaler, f)
    with open(METADATA_FILENAME, 'wb') as f: pickle.dump(metadata, f)

if __name__ == "__main__":
    main()