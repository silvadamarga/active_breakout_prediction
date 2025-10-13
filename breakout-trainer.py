#!/usr/bin/env python3
# train_breakout_tuning.py
"""
Hyperparameter-tuned, time-series-safe, multi-task breakout prediction model
with all enhancements and Walk-Forward Validation.

This script first finds the optimal hyperparameters on an initial data window,
then uses a walk-forward loop to backtest the full model strategy (with
market context, BiLSTM, and a dynamic target) over multiple time periods for
a robust evaluation. Results are saved to a file.
"""

import os
import logging
import pickle
import random
from datetime import datetime
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
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Conv1D, Dense, Dropout, LSTM,
                                     MaxPooling1D, Bidirectional)

# ----------------- Configuration -----------------
INITIAL_TICKERS = [
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM', 'V', 'JNJ',
    'UNH', 'HD', 'PG', 'MA', 'BAC', 'XOM', 'CVX', 'LLY', 'COST', 'PEP'
]
HISTORY_PERIOD = "7y"
TIME_STEPS = 60
BREAKOUT_PERIOD_DAYS = 15
MIN_HISTORY_REQUIREMENT = 252 * 2
MODEL_FILENAME = "breakout_tuned_model_wf_final.keras"
SCALER_FILENAME = "scaler_wf_final.pkl"
METADATA_FILENAME = "metadata_wf_final.pkl"
RESULTS_FILENAME = "backtest_results.txt" # File to save results
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS_PER_FOLD = 25
# KerasTuner specific configuration
TUNER_EPOCHS = 15
MAX_TRIALS = 20

# --- Walk-Forward Configuration ---
TRAIN_WINDOW_YEARS = 3
TEST_WINDOW_MONTHS = 6
STEP_MONTHS = 3

# ----------------- Logging -----------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# ----------------- Data Fetching & Initial Processing -----------------
def get_most_active_stocks(limit: int = 50) -> List[str]:
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=most_actives"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=8); resp.raise_for_status()
        data = resp.json()
        quotes = data.get("finance", {}).get("result", [{}])[0].get("quotes", [])
        tickers = [q["symbol"] for q in quotes if "symbol" in q]
        logging.info(f"Fetched {len(tickers)} most-active tickers from Yahoo.")
        return tickers[:limit]
    except Exception as e:
        logging.warning(f"Could not fetch most active tickers: {e}"); return []

def get_stock_data(ticker: str, period: str = HISTORY_PERIOD) -> Tuple[str, pd.DataFrame]:
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, auto_adjust=True)
        if df.empty: logging.warning(f"No data for {ticker}"); return ticker, None
        return ticker, df.rename_axis('Date')
    except Exception as e:
        logging.error(f"Error fetching {ticker}: {e}"); return ticker, None

def fetch_all_tickers(tickers: List[str], max_workers: int = 10) -> List[Tuple[str, pd.DataFrame]]:
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = [f.result() for f in [ex.submit(get_stock_data, t) for t in tickers]]
    valid_results = [(t, df) for t, df in results if df is not None]
    logging.info(f"Fetched valid data for {len(valid_results)}/{len(tickers)} tickers.")
    return valid_results

def process_data_for_ticker(ticker: str, df: pd.DataFrame, spy_df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) < MIN_HISTORY_REQUIREMENT: return None
    df = df.copy().sort_index()
    spy_df_renamed = spy_df.copy()[['Close']].rename(columns={'Close': 'SPY_Close'})
    df = df.merge(spy_df_renamed, left_index=True, right_index=True, how='left').ffill()
    close = df['Close']
    df['relative_strength'] = (df['Close'] / df['SPY_Close'])
    df['spy_sma_50_rel'] = (df['SPY_Close'] / df['SPY_Close'].rolling(50).mean()) - 1.0
    df['SMA_50_rel'] = (close / close.rolling(50).mean()) - 1.0
    df['ROC_20'] = close.pct_change(20)
    df['Volume_rel'] = (df['Volume'] / df['Volume'].rolling(20).mean()) - 1.0
    bb_length=20; bb_std=2.0
    rolling_mean = df['Close'].rolling(window=bb_length).mean()
    rolling_std = df['Close'].rolling(window=bb_length).std()
    df['BB_Width'] = ((rolling_mean + (rolling_std*bb_std)) - (rolling_mean - (rolling_std*bb_std)))/rolling_mean
    df.ta.obv(append=True)
    df['OBV_roc_20'] = df['OBV'].pct_change(20).replace([np.inf, -np.inf], np.nan)
    df.ta.rsi(length=14, append=True)
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    if isinstance(macd, pd.DataFrame) and 'MACD_12_26_9' in macd.columns:
        df['MACD_rel'] = macd['MACD_12_26_9'] / close
    else: df['MACD_rel'] = np.nan
    df.ta.adx(length=14, append=True)
    
    raw_atr = df.ta.atr(length=14)
    df['ATR_14_rel'] = raw_atr / close
    future_return = (close.shift(-BREAKOUT_PERIOD_DAYS) / close) - 1.0
    df['future_return_N'] = future_return * 100.0
    atr_multiplier = 3.0
    future_highs = df['High'].shift(-1).rolling(window=BREAKOUT_PERIOD_DAYS).max()
    breakout_level = df['Close'] + (raw_atr * atr_multiplier)
    df['breakout_next_N'] = (future_highs > breakout_level).astype(int)
    
    return df.dropna()

# ----------------- Sequence Creation (Refactored for folds) -----------------
def create_sequences_for_fold(df_fold: pd.DataFrame, features: List[str], time_steps: int):
    X, y_bk, y_ret = [], [], []
    for ticker, group in df_fold.groupby('ticker'):
        data_vals = group[features].values
        bk_labels = group['breakout_next_N'].values
        ret_labels = group['future_return_N'].values.clip(-50, 100)
        n = len(group)
        if n <= time_steps: continue
        for i in range(time_steps, n):
            X.append(data_vals[i - time_steps:i])
            y_bk.append(bk_labels[i])
            y_ret.append(ret_labels[i])
    return np.array(X), np.array(y_bk), np.array(y_ret)

# ----------------- Model Builder -----------------
def build_model(hp):
    n_timesteps = TIME_STEPS
    features = [
        'SMA_50_rel', 'ROC_20', 'Volume_rel', 'BB_Width', 'OBV_roc_20',
        'RSI_14', 'MACD_rel', 'ADX_14', 'ATR_14_rel',
        'relative_strength', 'spy_sma_50_rel'
    ]
    n_features = len(features)
    inp = Input(shape=(n_timesteps, n_features), name="input_seq")
    
    is_tuning = isinstance(hp, kt.HyperParameters)
    conv_filters = hp.Int('conv_filters', 32, 96, 32) if is_tuning else hp['conv_filters']
    lstm_units = hp.Int('lstm_units', 32, 96, 32) if is_tuning else hp['lstm_units']
    shared_units = hp.Int('shared_units', 32, 96, 32) if is_tuning else hp['shared_units']
    head_units = hp.Int('head_units', 16, 48, 16) if is_tuning else hp['head_units']
    dropout = hp.Float('dropout', 0.2, 0.5, 0.1) if is_tuning else hp['dropout']
    lr = hp.Choice('learning_rate', [1e-3, 5e-4, 1e-4]) if is_tuning else hp['learning_rate']

    x = Conv1D(filters=conv_filters, kernel_size=5, activation='relu', padding='same')(inp)
    x = MaxPooling1D(pool_size=2)(x)
    x = Bidirectional(LSTM(units=lstm_units, return_sequences=False, dropout=dropout, recurrent_dropout=dropout))(x)
    x = Dropout(dropout)(x)
    shared_trunk = Dense(units=shared_units, activation='relu', name='shared_trunk')(x)
    bk_head = Dense(units=head_units, activation='relu')(shared_trunk); bk_head = Dropout(dropout)(bk_head)
    breakout_output = Dense(1, activation='sigmoid', name='breakout')(bk_head)
    ret_head = Dense(units=head_units, activation='relu')(shared_trunk); ret_head = Dropout(dropout)(ret_head)
    return_output = Dense(1, activation='linear', name='return')(ret_head)
    model = Model(inputs=inp, outputs=[breakout_output, return_output])
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss={'breakout': 'binary_crossentropy', 'return': 'mean_squared_error'},
                  loss_weights={'breakout': 1.2, 'return': 1.0},
                  metrics={'breakout': [tf.keras.metrics.AUC(name='auc')], 'return': ['mae']})
    return model

# ----------------- Main Pipeline -----------------
def main():
    tf.random.set_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE); random.seed(RANDOM_STATE)
    logging.info("🚀 Initializing Walk-Forward Validation pipeline...")

    # --- 1. Data Fetching and Global Processing ---
    most_active = get_most_active_stocks(limit=30)
    tickers = sorted(list(set(INITIAL_TICKERS + most_active)))
    if 'SPY' not in tickers: tickers.append('SPY')
    ticker_dfs = fetch_all_tickers(tickers, max_workers=10)
    spy_df = next((df for t, df in ticker_dfs if t == 'SPY'), None)
    if spy_df is None: logging.error("Fatal: Could not fetch SPY data."); return
    ticker_dfs = [(t, df) for t, df in ticker_dfs if t != 'SPY']
    features = [
        'SMA_50_rel', 'ROC_20', 'Volume_rel', 'BB_Width', 'OBV_roc_20', 'RSI_14',
        'MACD_rel', 'ADX_14', 'ATR_14_rel', 'relative_strength', 'spy_sma_50_rel'
    ]
    processed_dfs = [process_data_for_ticker(t, df, spy_df) for t, df in ticker_dfs]
    full_dataset = pd.concat([df.assign(ticker=t) for (t, _), df in zip(ticker_dfs, processed_dfs) if df is not None])
    full_dataset['Date'] = full_dataset.index.tz_localize(None)
    
    all_dates = sorted(full_dataset['Date'].unique())
    
    # --- 2. Hyperparameter Search on the First Fold ---
    logging.info("⚙️ Finding best hyperparameters on the first data fold...")
    train_end_date = all_dates[0] + pd.DateOffset(years=TRAIN_WINDOW_YEARS)
    val_end_date = train_end_date + pd.DateOffset(months=TEST_WINDOW_MONTHS)
    
    train_val_df = full_dataset[full_dataset['Date'] < val_end_date]
    train_df = train_val_df[train_val_df['Date'] < train_end_date]
    val_df = train_val_df[train_val_df['Date'] >= train_end_date]

    X_train, y_bk_train, y_ret_train = create_sequences_for_fold(train_df, features, TIME_STEPS)
    X_val, y_bk_val, y_ret_val = create_sequences_for_fold(val_df, features, TIME_STEPS)
    
    scaler = MinMaxScaler(); scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    X_train_scaled = scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

    tuner = kt.Hyperband(build_model, objective=kt.Objective("val_breakout_auc", direction="max"),
                         max_epochs=TUNER_EPOCHS, factor=3, directory='keras_tuner_dir',
                         project_name='breakout_tuning_ultimate', overwrite=True)
    tuner.search(X_train_scaled, [y_bk_train, y_ret_train], validation_data=(X_val_scaled, [y_bk_val, y_ret_val]),
                 epochs=TUNER_EPOCHS, batch_size=BATCH_SIZE, callbacks=[EarlyStopping('val_loss', patience=5)])
    
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    logging.info(f"🏆 Best Hyperparameters Found: \n {best_hps.values}")

    # --- 3. Walk-Forward Validation Loop ---
    oos_preds_bk, oos_preds_ret = [], []
    oos_actuals_bk, oos_actuals_ret = [], []
    
    step_offset = pd.DateOffset(months=STEP_MONTHS)
    train_offset = pd.DateOffset(years=TRAIN_WINDOW_YEARS)
    test_offset = pd.DateOffset(months=TEST_WINDOW_MONTHS)
    
    start_date = all_dates[0]
    end_date = all_dates[-1] - test_offset - train_offset
    
    current_date = start_date
    fold = 1
    while current_date <= end_date:
        train_start = current_date
        train_end = train_start + train_offset
        test_end = train_end + test_offset
        logging.info(f"--- FOLD {fold}: Training on {train_start.date()}->{train_end.date()}, Testing on {train_end.date()}->{test_end.date()} ---")

        train_df = full_dataset[(full_dataset['Date'] >= train_start) & (full_dataset['Date'] < train_end)]
        test_df = full_dataset[(full_dataset['Date'] >= train_end) & (full_dataset['Date'] < test_end)]
        
        X_train, y_bk_train, y_ret_train = create_sequences_for_fold(train_df, features, TIME_STEPS)
        X_test, y_bk_test, y_ret_test = create_sequences_for_fold(test_df, features, TIME_STEPS)
        
        if len(X_train) == 0 or len(X_test) == 0:
            logging.warning(f"Skipping Fold {fold} due to insufficient data.")
            current_date += step_offset; fold += 1; continue
            
        scaler = MinMaxScaler(); scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
        X_train_scaled = scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        unique, _ = np.unique(y_bk_train, return_counts=True)
        if len(unique) > 1:
            class_weights = compute_class_weight('balanced', classes=unique, y=y_bk_train)
            weight_map = dict(zip(unique, class_weights))
            breakout_sample_weights = np.array([weight_map[label] for label in y_bk_train])
        else: breakout_sample_weights = np.ones_like(y_bk_train, dtype=float)

        model = build_model(best_hps)
        model.fit(X_train_scaled, [y_bk_train, y_ret_train], epochs=EPOCHS_PER_FOLD, batch_size=BATCH_SIZE, verbose=2,
                  sample_weight=[breakout_sample_weights, np.ones_like(y_ret_train, dtype=float)])
        
        pred_bk, pred_ret = model.predict(X_test_scaled)
        oos_preds_bk.extend(pred_bk.ravel()); oos_preds_ret.extend(pred_ret.ravel())
        oos_actuals_bk.extend(y_bk_test); oos_actuals_ret.extend(y_ret_test)

        current_date += step_offset; fold += 1

    # --- 4. Final Aggregated Evaluation ---
    logging.info("="*20 + " FINAL WALK-FORWARD METRICS " + "="*20)
    if len(oos_actuals_bk) > 0:
        auc_bk = roc_auc_score(oos_actuals_bk, oos_preds_bk)
        ap_bk = average_precision_score(oos_actuals_bk, oos_preds_bk)
        mae_ret = mean_absolute_error(oos_actuals_ret, oos_preds_ret)
        rmse_ret = np.sqrt(mean_squared_error(oos_actuals_ret, oos_preds_ret))
        logging.info(f"BREAKOUT -> AUC: {auc_bk:.4f} | PR-AUC: {ap_bk:.4f}")
        logging.info(f"RETURN   -> MAE: {mae_ret:.2f}% | RMSE: {rmse_ret:.2f}%")

        # ### NEW: SAVE RESULTS TO FILE ###
        logging.info(f"💾 Saving results to {RESULTS_FILENAME}...")
        with open(RESULTS_FILENAME, 'a') as f:
            f.write("="*65 + "\n")
            f.write(f"Run Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Best Hyperparameters: {best_hps.values}\n")
            f.write(f"BREAKOUT -> AUC: {auc_bk:.4f} | PR-AUC: {ap_bk:.4f}\n")
            f.write(f"RETURN   -> MAE: {mae_ret:.2f}% | RMSE: {rmse_ret:.2f}%\n")
            f.write("="*65 + "\n\n")

    else:
        logging.warning("No out-of-sample predictions were generated.")
    logging.info("="*65)

    # --- 5. Retrain on all data and save final artifacts ---
    logging.info("🏋️ Retraining final model on all available data...")
    X_all, y_bk_all, y_ret_all = create_sequences_for_fold(full_dataset, features, TIME_STEPS)
    scaler_final = MinMaxScaler().fit(X_all.reshape(-1, X_all.shape[-1]))
    X_all_scaled = scaler_final.transform(X_all.reshape(-1, X_all.shape[-1])).reshape(X_all.shape)

    final_model = build_model(best_hps)
    unique_all, _ = np.unique(y_bk_all, return_counts=True)
    if len(unique_all) > 1:
        class_weights_all = compute_class_weight('balanced', classes=unique_all, y=y_bk_all)
        weight_map_all = dict(zip(unique_all, class_weights_all))
        final_sample_weights = np.array([weight_map_all[label] for label in y_bk_all])
    else: final_sample_weights = np.ones_like(y_bk_all, dtype=float)
        
    final_model.fit(X_all_scaled, [y_bk_all, y_ret_all], epochs=EPOCHS_PER_FOLD, batch_size=BATCH_SIZE, verbose=2,
                    sample_weight=[final_sample_weights, np.ones_like(y_ret_all, dtype=float)])

    logging.info("💾 Saving final model, scaler, and metadata...")
    final_model.save(MODEL_FILENAME)
    metadata = {'features': features, 'used_tickers': [t for t, df in ticker_dfs]}
    with open(SCALER_FILENAME, 'wb') as f: pickle.dump(scaler_final, f)
    with open(METADATA_FILENAME, 'wb') as f: pickle.dump(metadata, f)

if __name__ == "__main__":
    main()