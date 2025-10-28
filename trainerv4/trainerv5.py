#!/usr/bin/env python3
# trainerv5_combined_final.py (v6 - Improved Strategy)
"""
Master model training script with strategy improvements based on analysis.

--- Strategy Changes (v6) ---
1.  Stricter Regime Filter: Default entry only in "Low Risk" (Regime 0).
    Configurable via --entry-regimes argument.
2.  Higher Confidence Threshold: Default increased to 0.75.
3.  Combined Exit Strategy:
    - ATR-based initial stop (default 2*ATR).
    - ATR-based trailing stop (default 1.5*ATR).
    - Max holding period time exit re-enabled.
---------------------------------
"""

import os
import logging
import pickle
import random
import time
import json
import argparse
import sqlite3
import gc # Import garbage collector for memory management
from datetime import datetime
from io import StringIO
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict

# Suppress excessive TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import pandas_ta as ta
import tensorflow as tf
import keras_tuner as kt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, mean_squared_error, classification_report

from tensorflow.keras import Input, Model, optimizers, backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Conv1D, Dense, Dropout, LSTM, GRU, MaxPooling1D, Bidirectional,
    GlobalAveragePooling1D, LayerNormalization, Masking
)
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

# ---------------- Config ----------------
HISTORY_PERIOD_YEARS = 15
MIN_HISTORY_REQUIREMENT = 252 * 3
TIME_STEPS = 60 # Used for both regime and breakout models
BREAKOUT_PERIOD_DAYS = 15
RANDOM_STATE = 42
BATCH_SIZE = 16 # Keep reduced batch size
CACHE_DIR = "data_cache_combined"
DB_CACHE_FILE = os.path.join(CACHE_DIR, "market_data_cache_v5.db")
TICKERS_JSON_FILE = "tickers.json" # Source for custom tickers

# --- Breakout Model (Trainer) Config ---
RESULTS_FILENAME_TPL = "backtests/backtest_results_{run_id}.txt"
TRADES_FILENAME_TPL = "backtests/backtest_trades_{run_id}.csv"
PREDICTIONS_FILENAME_TPL = "backtests/oos_predictions_{run_id}.csv"
EQUITY_CURVE_FILENAME_TPL = "backtests/equity_curve_{run_id}.png"
JSON_SUMMARY_FILENAME_TPL = "backtests/summary_{run_id}.json"
LOG_FILENAME_TPL = "backtests/run_log_{run_id}.log"
PORTFOLIO_FOLD_CACHE_TPL = "backtests/pf_fold_{run_id}_{fold_id}.pkl"


BREAKOUT_MODEL_FILENAME = "final_tuned_model_v5_final.keras"
BREAKOUT_SCALER_FILENAME = "final_scaler_v5_final.pkl"
BREAKOUT_METADATA_FILENAME = "final_metadata_v5_final.pkl"

TUNER_EPOCHS = 15
MAX_TRIALS = 35
EPOCHS_PER_FOLD = 4
TUNER_BATCH_SIZE = 16

# --- Regime Model Config ---
REGIME_MODEL_FILENAME = "market_regime_model_v4.keras"
REGIME_SCALER_FILENAME = "market_regime_scaler_v4.pkl"
REGIME_METADATA_FILENAME = "market_regime_metadata_v4.pkl"
# BULLISH_REGIMES no longer used directly for entry filtering, see ALLOWED_ENTRY_REGIMES
REGIME_TICKERS = ["SPY", "^VIX", "^TNX", "^FVX", "HYG", "IEF", "GLD"]

# --- Backtest Strategy Config (v6 Updates) ---
INITIAL_PORTFOLIO_CASH = 100000.0
DEFAULT_STRATEGY_PARAMS = {
    'confidence_threshold': 0.76, # Increased default
    'initial_stop_atr_multiplier': 2.1, # ATR multiplier for initial stop
    'trailing_stop_atr_multiplier': 1.6, # ATR multiplier for trailing stop
    'holding_period_days': 15, # Max holding period
    'transaction_cost_percent': 0.0005,
    'slippage_percent': 0.0005,
    'risk_per_trade_percent': 0.01,
    'allowed_entry_regimes': [0, 1]
}

# --- Other Config ---
SECTOR_ETF_MAP = { # Unchanged
    'Technology': 'XLK', 'Financials': 'XLF', 'Health Care': 'XLV',
    'Consumer Discretionary': 'XLY', 'Industrials': 'XLI', 'Consumer Staples': 'XLP',
    'Energy': 'XLE', 'Utilities': 'XLU', 'Real Estate': 'XLRE',
    'Materials': 'XLB', 'Communication Services': 'XLC'
}
REGIME_NAME_MAP = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk", 3: "Crisis"} # For analysis

# --- Checkpoint Cache Filenames --- (Unchanged)
SP500_TICKERS_CACHE = os.path.join(CACHE_DIR, "sp500_tickers.json")
SECTOR_MAP_CACHE = os.path.join(CACHE_DIR, "ticker_to_sector.json")
REGIME_PREDS_CACHE = os.path.join(CACHE_DIR, "checkpoint_predicted_regimes.pkl")
BREAKOUT_DATASET_CACHE = os.path.join(CACHE_DIR, "checkpoint_full_breakout_dataset.pkl")
BREAKOUT_FEATURES_CACHE = os.path.join(CACHE_DIR, "checkpoint_full_breakout_features.json")
BEST_HPS_CACHE = os.path.join(CACHE_DIR, "checkpoint_best_hps.json")


logging.getLogger().setLevel(logging.INFO)
tf.get_logger().setLevel('ERROR')


# ---------------- DataManager (Offline-Capable) ----------------
# (Class definition is unchanged)
class DataManager:
    def __init__(self, db_path: str, offline_mode: bool = False):
        self.db_path = db_path
        self.offline_mode = offline_mode
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._create_table()

    def _create_table(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stock_data (
                    ticker TEXT PRIMARY KEY,
                    fetch_date TEXT,
                    data BLOB
                )
            """)

    def get_stock_data(self, ticker: str) -> Tuple[str, pd.DataFrame]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT data FROM stock_data WHERE ticker=?", (ticker,))
                result = cursor.fetchone()
                if result:
                    return ticker, pickle.loads(result[0])
        except Exception as e:
            logging.warning(f"Error reading {ticker} from SQLite cache: {e}")

        if self.offline_mode:
            logging.warning(f"Offline mode: Cache miss for {ticker}. Returning None.")
            return ticker, None

        logging.info(f"Online mode: Cache miss for {ticker}, fetching from yfinance...")
        max_retries = 3
        backoff_factor = 2
        for i in range(max_retries):
            try:
                df = yf.Ticker(ticker).history(period=f"{HISTORY_PERIOD_YEARS}y", auto_adjust=True)
                if df.empty:
                    logging.warning(f"No data for {ticker} from yfinance.")
                    return ticker, None

                df.index = df.index.tz_localize(None)
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("INSERT OR REPLACE INTO stock_data (ticker, fetch_date, data) VALUES (?, ?, ?)",
                                 (ticker, datetime.now().strftime('%Y-%m-%d'), pickle.dumps(df)))
                return ticker, df.rename_axis('Date')

            except Exception as e:
                wait_time = backoff_factor ** i
                logging.warning(f"Error fetching {ticker}: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

        logging.error(f"Failed to fetch {ticker} after {max_retries} retries.")
        return ticker, None

    def fetch_all_data(self, tickers: List[str], max_workers: int = 12) -> Dict[str, pd.DataFrame]:
        logging.info(f"DataManager fetching/loading {len(tickers)} tickers...")
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(self.get_stock_data, tickers))

        valid_data = {t: df for t, df in results if df is not None and not df.empty}

        if self.offline_mode:
            logging.info(f"Offline load complete for {len(valid_data)}/{len(tickers)} tickers.")
        else:
            logging.info(f"Online fetch/load complete for {len(valid_data)}/{len(tickers)} tickers.")
        return valid_data

# ---------------- Data Sync Phase Functions ----------------
# (Functions get_sp500_tickers_cached, get_tickers_from_json,
#  get_or_create_sector_map, run_data_sync_phase are unchanged)
def get_sp500_tickers_cached(offline_mode: bool = False) -> list:
    if os.path.exists(SP500_TICKERS_CACHE):
        logging.info(f"Loading S&P 500 tickers from local cache: {SP500_TICKERS_CACHE}")
        with open(SP500_TICKERS_CACHE, 'r') as f:
            return json.load(f)

    if offline_mode:
        logging.error(f"Offline mode: S&P 500 ticker cache not found at {SP500_TICKERS_CACHE}. Aborting.")
        return []

    logging.info("Online mode: Scraping S&P 500 tickers from Wikipedia...")
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=10); r.raise_for_status()
        table = pd.read_html(StringIO(r.text))
        tickers = table[0]['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers] # Fix for 'BRK.B'

        logging.info(f"Scraped {len(tickers)} S&P 500 tickers. Caching to {SP500_TICKERS_CACHE}")
        with open(SP500_TICKERS_CACHE, 'w') as f:
            json.dump(tickers, f)
        return tickers
    except Exception as e:
        logging.error(f"Could not scrape S&P500 tickers: {e}")
        return []

def get_tickers_from_json(filename: str) -> list:
    if not os.path.exists(filename):
        logging.warning(f"'{filename}' not found. Skipping custom tickers.")
        return []
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict):
                tickers = list(data.keys())
            elif isinstance(data, list):
                tickers = data
            else:
                raise ValueError("JSON file is not a dict or list")
        logging.info(f"Loaded {len(tickers)} tickers from '{filename}'.")
        return tickers
    except Exception as e:
        logging.error(f"Could not read or parse '{filename}': {e}")
        return []

def get_or_create_sector_map(tickers: List[str], offline_mode: bool = False) -> dict:
    if os.path.exists(SECTOR_MAP_CACHE):
        logging.info(f"Loading ticker-to-sector map from local cache: {SECTOR_MAP_CACHE}")
        with open(SECTOR_MAP_CACHE, 'r') as f:
            return json.load(f)

    if offline_mode:
        logging.error(f"Offline mode: Sector map cache not found at {SECTOR_MAP_CACHE}. Aborting.")
        return {}

    logging.info(f"Online mode: Fetching sector info for {len(tickers)} tickers (this may take a long time)...")
    ticker_to_sector = {}
    for i, t in enumerate(tickers):
        if i % 50 == 0:
            logging.info(f"Fetching sector... {i}/{len(tickers)}")
        try:
            info = yf.Ticker(t).info
            sec = info.get('sector')
            if sec:
                ticker_to_sector[t] = sec
        except Exception:
            time.sleep(0.1)
            continue

    logging.info(f"Fetched sector info for {len(ticker_to_sector)} tickers. Caching to {SECTOR_MAP_CACHE}")
    with open(SECTOR_MAP_CACHE, 'w') as f:
        json.dump(ticker_to_sector, f)
    return ticker_to_sector

def run_data_sync_phase(data_manager: DataManager, offline_mode: bool = False) -> Tuple[list, dict]:
    logging.info(f"--- Starting Data Sync Phase (Offline Mode: {offline_mode}) ---")

    tickers_sp = get_sp500_tickers_cached(offline_mode)
    custom_tickers = get_tickers_from_json(TICKERS_JSON_FILE)
    base_tickers = sorted(list(set(tickers_sp + custom_tickers)))
    if not base_tickers:
        logging.error("No base tickers found. Aborting.")
        return [], {}

    ticker_to_sector = get_or_create_sector_map(base_tickers, offline_mode)
    if not ticker_to_sector and offline_mode:
        logging.error("Could not load sector map in offline mode. Aborting.")
        return [], {}

    sector_etfs = list(set(SECTOR_ETF_MAP.values()))
    all_tickers_to_fetch = list(set(base_tickers + sector_etfs + REGIME_TICKERS))

    data_manager.fetch_all_data(all_tickers_to_fetch)

    logging.info("--- Data Sync Phase Complete ---")
    return base_tickers, ticker_to_sector


# ---------------- Regime Model: Check/Load/Predict ----------------

# --- REFACTORED: Only Checks Existence ---
def train_or_load_regime_model(data_manager: DataManager) -> bool:
    """
    Checks if all required regime model artifacts exist.
    Returns True if found, False otherwise.
    (Does NOT train the model anymore).
    """
    required_files = [REGIME_MODEL_FILENAME, REGIME_SCALER_FILENAME, REGIME_METADATA_FILENAME]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if not missing_files:
        logging.info("All regime model artifacts found.")
        return True
    else:
        logging.error(f"Missing required regime model artifacts: {', '.join(missing_files)}")
        logging.error("This script no longer trains the regime model. Please ensure the files exist or run the regime trainer script first.")
        return False

# --- KEPT: Needed for Prediction ---
def create_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    df['SMA_50_rel'] = (df['Close'] / df['Close'].rolling(50).mean()) - 1.0
    df['SMA_200_rel'] = (df['Close'] / df['Close'].rolling(200).mean()) - 1.0
    try:
        df.ta.rsi(length=14, append=True)
    except Exception as e:
        logging.warning(f"Could not calculate RSI for regime features: {e}")
        df['RSI_14'] = np.nan

    df['RSI_14_mom'] = df['RSI_14'].diff(5) if 'RSI_14' in df else np.nan
    df['yield_curve_mom'] = df['yield_curve_5y10y'].diff(5)

    if 'IEF_Close' in df and df['IEF_Close'].notna().any():
        df['spy_vs_bonds'] = (df['Close'] / df['IEF_Close']).pct_change(20)
    else:
        df['spy_vs_bonds'] = np.nan
    if 'GLD_Close' in df and df['GLD_Close'].notna().any():
        df['spy_vs_gold'] = (df['Close'] / df['GLD_Close']).pct_change(20)
    else:
        df['spy_vs_gold'] = np.nan

    cols_to_drop = ['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits']
    # If the regime model NEEDS 'Close', remove 'Close' from cols_to_drop.
    # cols_to_drop.append('Close')
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    return df

# --- Load and Predict Functions Unchanged ---
def load_regime_model_and_artifacts():
    try:
        logging.info("Loading pre-trained market regime model and artifacts...")
        model = tf.keras.models.load_model(REGIME_MODEL_FILENAME, safe_mode=False)
        with open(REGIME_SCALER_FILENAME, 'rb') as f: scaler = pickle.load(f)
        with open(REGIME_METADATA_FILENAME, 'rb') as f: metadata = pickle.load(f)
        logging.info("Regime model loaded successfully.")
        return model, scaler, metadata['features'], metadata['regime_map']
    except Exception as e:
        logging.error(f"FATAL: Could not load regime model at {REGIME_MODEL_FILENAME}. Error: {e}")
        return None, None, None, None

def predict_regimes_for_period(market_df: pd.DataFrame, model, scaler, features: list) -> pd.DataFrame:
    logging.info("Generating daily market regime predictions for the full period...")
    missing_features = [f for f in features if f not in market_df.columns]
    if missing_features:
        logging.error(f"Missing required features for regime prediction: {', '.join(missing_features)}")
        return pd.DataFrame()

    X, dates = [], []
    data = market_df.loc[:, features].values
    for i in range(TIME_STEPS, len(market_df)):
        X.append(data[i-TIME_STEPS:i]); dates.append(market_df.index[i])

    if not X:
        logging.error("No sequences generated for regime prediction.")
        return pd.DataFrame()

    X = np.array(X)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        X_scaled = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    except ValueError as e:
         logging.error(f"Error during regime scaler transformation: {e}")
         return pd.DataFrame()

    predictions = np.argmax(model.predict(X_scaled, batch_size=BATCH_SIZE*4), axis=1)
    return pd.DataFrame({'predicted_regime': predictions}, index=pd.to_datetime(dates))

# ---------------- Breakout Model: Feature Helper ----------------
# (process_data_for_ticker function unchanged from previous version)
def process_data_for_ticker(ticker: str, df: pd.DataFrame, context_dfs: dict, ticker_to_sector: dict, strategy_params: dict) -> pd.DataFrame:
    if df is None or len(df) < MIN_HISTORY_REQUIREMENT: return None
    df = df.copy().sort_index()

    if 'predicted_regimes' in context_dfs:
        df = df.merge(context_dfs['predicted_regimes'], left_index=True, right_index=True, how='left')
        df['predicted_regime'] = df['predicted_regime'].ffill()

    if 'SPY' in context_dfs:
        df = df.merge(context_dfs['SPY'][['SPY_Close']], left_index=True, right_index=True, how='left')

    sector = ticker_to_sector.get(ticker)
    if sector and sector in SECTOR_ETF_MAP:
        etf = SECTOR_ETF_MAP[sector]
        secdf = context_dfs.get(etf)
        if secdf is not None:
            df = df.merge(secdf[['Sector_Close','sector_sma_50_rel']], left_index=True, right_index=True, how='left')

    df.ffill(inplace=True)
    if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
        logging.warning(f"Skipping {ticker} due to missing OHLCV columns.")
        return None
    close = df['Close']

    df['SMA_50_rel'] = (close / close.rolling(50).mean()) - 1.0
    df['ROC_20'] = close.pct_change(20)
    vol_mean = df['Volume'].rolling(20).mean()
    df['Volume_rel'] = (df['Volume'] / vol_mean.replace(0, np.nan)) - 1.0

    try:
        df.ta.rsi(length=14, append=True)
        df.ta.adx(length=14, append=True)
        df.ta.obv(append=True)
        df.ta.cmf(length=20, append=True)
        stoch = df.ta.stoch(k=14, d=3, smooth_k=3)
        if isinstance(stoch, pd.DataFrame):
            df['STOCHk_14_3_3'] = stoch['STOCHk_14_3_3']
            df['STOCHd_14_3_3'] = stoch['STOCHd_14_3_3']

        if 'Close' in df and df['Close'].notna().any():
             safe_close = df['Close'].replace(to_replace=0, method='ffill').replace(to_replace=lambda x: x < 0, value=np.nan)
             log_return = np.log(safe_close / safe_close.shift(1))
             df['historical_vol_20'] = log_return.rolling(window=20).std() * np.sqrt(252)
        else:
             df['historical_vol_20'] = np.nan

        raw_atr = df.ta.atr(length=14)
        if isinstance(raw_atr, pd.DataFrame): raw_atr = raw_atr.iloc[:, 0]

    except Exception as e:
        logging.warning(f"TA-Lib error on {ticker}: {e}. Using simple ATR/Vol.")
        df['historical_vol_20'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
        raw_atr = (df['High'] - df['Low']).rolling(14).mean()

    df['raw_atr'] = raw_atr
    df['ATR_14_rel'] = raw_atr / close.replace(0, np.nan)
    spy_close = df.get('SPY_Close', close).replace(0, np.nan)
    df['relative_strength_market'] = df['Close'] / spy_close
    if 'Sector_Close' in df.columns:
        sector_close = df['Sector_Close'].replace(0, np.nan)
        df['relative_strength_sector'] = df['Close'] / sector_close

    idx = df.index
    df['month_sin'] = np.sin(2*np.pi*idx.month/12)
    df['month_cos'] = np.cos(2*np.pi*idx.month/12)

    future_price = close.shift(-BREAKOUT_PERIOD_DAYS)
    safe_close_for_log = close.replace(0, np.nan)
    safe_future_price_for_log = future_price.replace(0, np.nan)
    log_ret = np.log(safe_future_price_for_log / safe_close_for_log)
    df['future_return_N'] = log_ret * 100.0

    future_highs = df['High'].shift(-1).rolling(window=BREAKOUT_PERIOD_DAYS).max()
    # Use global FEATURES list if defined, otherwise guess essential cols
    global FEATURES # Make sure FEATURES is accessible
    essential_cols_default = ['SMA_50_rel', 'ROC_20'] # Example, update if needed
    current_features = FEATURES if 'FEATURES' in globals() and FEATURES else essential_cols_default

    # Calculate breakout level using the atr_multiplier from strategy_params
    breakout_level = df['Close'] + (df['raw_atr'].fillna(0) * strategy_params.get('atr_multiplier', 3.0))
    df['breakout_next_N'] = (future_highs > breakout_level).astype(int)


    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    essential_cols = current_features + ['breakout_next_N', 'future_return_N']
    df.dropna(subset=[col for col in essential_cols if col in df.columns], inplace=True)

    return df if len(df) >= MIN_HISTORY_REQUIREMENT else None


# ---------------- Purged K-Fold CV (Corrected) ----------------
# (Class definition is unchanged)
class PurgedKFold(KFold):
    def __init__(self, n_splits=5, t1=None, pctEmbargo=0.01):
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pctEmbargo = pctEmbargo

    def split(self, X, y=None, groups=None):
        if X.size == 0:
            raise ValueError("Input X cannot be empty")

        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0] * self.pctEmbargo)
        test_starts = [(i[0], i[-1] + 1) for i in np.array_split(indices, self.n_splits)]

        for i, j in test_starts:
            if i >= len(self.t1): continue
            t0 = self.t1.iloc[i]
            test_indices = indices[i:j]
            if j < len(self.t1):
                t1 = self.t1.iloc[j]
            else:
                t1 = self.t1.iloc[-1]

            train_indices_embargoed = np.concatenate([
                indices[:i],
                indices[j + mbrg:]
            ])

            if self.t1 is not None:
                train_t1 = self.t1.iloc[train_indices_embargoed]
                train_indices = train_indices_embargoed[
                    (train_t1 < t0) | (train_t1 > t1)
                ]
            else:
                train_indices = train_indices_embargoed

            yield train_indices, test_indices

# ---------------- Breakout Model: Sequence & Model Builders ----------------
# (Functions create_breakout_sequences, build_model_from_hp, build_model_from_dict unchanged)
def create_breakout_sequences(df_fold: pd.DataFrame, features: List[str], time_steps: int):
    X,y_bk,y_ret,seq_info = [],[],[],[]
    for ticker, group in df_fold.groupby('ticker'):
        data = group[features].values.astype(np.float32)
        bk = group['breakout_next_N'].values
        ret = np.nan_to_num(
            group['future_return_N'].values, nan=0.0, posinf=100.0, neginf=-50.0
        ).clip(-50, 100)

        n = len(group)
        if n <= time_steps: continue
        for i in range(time_steps, n):
            X.append(data[i-time_steps:i])
            y_bk.append(bk[i])
            y_ret.append(ret[i])
            seq_info.append({'ticker': ticker, 'date': group.index[i]})
    return np.array(X, dtype=np.float32), np.array(y_bk), np.array(y_ret), seq_info

def build_model_from_hp(hp, n_features):
    """Builds model for Keras Tuner."""
    conv_filters_1 = hp.Int('conv_filters_1', 32, 128, step=32, default=64)
    conv_filters_2 = hp.Int('conv_filters_2', 32, 64, step=16, default=48)
    kernel_size = hp.Choice('kernel_size', [3, 5, 7], default=5)
    rnn_type = hp.Choice('rnn_type', ['lstm', 'gru'], default='gru')
    rnn_units = hp.Int('rnn_units', 32, 128, step=32, default=64)
    shared_units = hp.Int('shared_units', 32, 96, step=32, default=64)
    head_units = hp.Int('head_units', 16, 64, step=16, default=32)
    dropout = hp.Float('dropout', 0.1, 0.4, step=0.05, default=0.2)
    recurrent_dropout = 0.2
    lr = hp.Choice('learning_rate', [1e-3, 5e-4, 1e-4], default=1e-3)
    use_attention = hp.Boolean('use_attention', default=True)

    inp = Input(shape=(TIME_STEPS, n_features), name='inp')
    x = Conv1D(filters=conv_filters_1, kernel_size=kernel_size, activation='relu', padding='same')(inp)
    x = Conv1D(filters=conv_filters_2, kernel_size=kernel_size, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = LayerNormalization()(x)

    if rnn_type == 'lstm':
        rnn_layer = LSTM(units=rnn_units, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout)
    else:
        rnn_layer = GRU(units=rnn_units, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout, reset_after=False)

    x = Bidirectional(rnn_layer)(x)
    x = LayerNormalization()(x)

    if use_attention:
        score = Dense(1, activation='tanh')(x)
        score = layers.Softmax(axis=1)(score)
        x = layers.Multiply()([x, score])
        x = GlobalAveragePooling1D()(x)
    else:
        x = GlobalAveragePooling1D()(x)

    x = Dropout(dropout)(x)
    shared = Dense(shared_units, activation='relu')(x)
    shared = LayerNormalization()(shared)

    bk = Dense(head_units, activation='relu')(shared)
    bk = Dropout(dropout)(bk)
    breakout_output = Dense(1, activation='sigmoid', name='breakout')(bk)

    ret = Dense(head_units, activation='relu')(shared)
    ret = Dropout(dropout)(ret)
    return_output = Dense(1, activation='linear', name='return')(ret)

    model = Model(inputs=inp, outputs=[breakout_output, return_output])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss={'breakout': 'binary_crossentropy', 'return': tf.keras.losses.Huber()},
        loss_weights={'breakout': 1.0, 'return': 0.5},
        metrics={'breakout': [tf.keras.metrics.AUC(name='auc')], 'return': ['mae']}
    )
    return model

def build_model_from_dict(hp_values: dict, n_features: int):
    conv_filters_1 = int(hp_values.get('conv_filters_1', 64))
    conv_filters_2 = int(hp_values.get('conv_filters_2', 48))
    kernel_size = int(hp_values.get('kernel_size', 5))
    rnn_type = hp_values.get('rnn_type', 'gru')
    rnn_units = int(hp_values.get('rnn_units', 64))
    shared_units = int(hp_values.get('shared_units', 64))
    head_units = int(hp_values.get('head_units', 32))
    dropout = float(hp_values.get('dropout', 0.2))
    recurrent_dropout = 0.2
    lr = float(hp_values.get('learning_rate', 1e-3))
    use_attention = bool(hp_values.get('use_attention', True))

    inp = Input(shape=(TIME_STEPS, n_features), name='inp')
    x = Conv1D(filters=conv_filters_1, kernel_size=kernel_size, activation='relu', padding='same')(inp)
    x = Conv1D(filters=conv_filters_2, kernel_size=kernel_size, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = LayerNormalization()(x)

    if rnn_type == 'lstm':
        rnn_layer = LSTM(units=rnn_units, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout)
    else:
        rnn_layer = GRU(units=rnn_units, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout, reset_after=False)

    x = Bidirectional(rnn_layer)(x)
    x = LayerNormalization()(x)

    if use_attention:
        score = Dense(1, activation='tanh')(x)
        score = layers.Softmax(axis=1)(score)
        x = layers.Multiply()([x, score])
        x = GlobalAveragePooling1D()(x)
    else:
        x = GlobalAveragePooling1D()(x)

    x = Dropout(dropout)(x)
    shared = Dense(shared_units, activation='relu')(x)
    shared = LayerNormalization()(shared)

    bk = Dense(head_units, activation='relu')(shared)
    bk = Dropout(dropout)(bk)
    breakout_output = Dense(1, activation='sigmoid', name='breakout')(bk)

    ret = Dense(head_units, activation='relu')(shared)
    ret = Dropout(dropout)(ret)
    return_output = Dense(1, activation='linear', name='return')(ret)

    model = Model(inputs=inp, outputs=[breakout_output, return_output])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss={'breakout': 'binary_crossentropy', 'return': tf.keras.losses.Huber()},
        loss_weights={'breakout': 1.0, 'return': 0.5},
        metrics={'breakout': [tf.keras.metrics.AUC(name='auc')], 'return': ['mae']}
    )
    return model

# --- tf.data.Dataset Helper Function (Yields Weights Tuple) ---
def create_tf_dataset(X, y_bk, y_ret, breakout_sample_weights, batch_size, shuffle=False):
    """Wraps numpy arrays in a memory-efficient tf.data.Dataset."""
    targets = {'breakout': y_bk, 'return': y_ret}
    weights_tuple = (
        breakout_sample_weights,
        np.ones_like(y_ret, dtype=np.float32)
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (X, targets, weights_tuple)
    )
    if shuffle:
        buffer_size = min(len(X), 100000)
        dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# --- UPDATED Backtester Function (v6 Strategy) ---
def run_backtest_on_fold(predictions_df: pd.DataFrame, full_test_data: pd.DataFrame, strategy_params: dict, initial_cash: float):
    logging.info(f"Running backtest from {predictions_df['date'].min()} to {predictions_df['date'].max()} with {initial_cash:.2f} starting cash")
    cash = float(initial_cash)
    portfolio = {}
    trades = []
    portfolio_values = []

    # Extract strategy parameters
    cost_pct = strategy_params.get('transaction_cost_percent', 0.0005)
    slip_pct = strategy_params.get('slippage_percent', 0.0005)
    risk_per_trade = strategy_params.get('risk_per_trade_percent', 0.01)
    holding_period = strategy_params['holding_period_days']
    confidence_thresh = strategy_params['confidence_threshold']
    allowed_entry_regimes = strategy_params.get('allowed_entry_regimes', [0]) # Default to Low Risk only
    initial_stop_atr_mult = strategy_params['initial_stop_atr_multiplier']
    trailing_stop_atr_mult = strategy_params['trailing_stop_atr_multiplier']

    unique_dates = sorted(pd.to_datetime(predictions_df['date'].unique()))

    for current_date in unique_dates:
        # 1. Mark-to-Market
        current_portfolio_value = cash
        for t, info in portfolio.items():
            row = full_test_data[(full_test_data.index == current_date) & (full_test_data['ticker'] == t)]
            market_price = float(row['Close'].iloc[0]) if not row.empty else info['entry_price']
            current_portfolio_value += info['shares'] * market_price
        portfolio_values.append({'date': current_date, 'total_value': current_portfolio_value})

        # 2. Process Exits (Stop-Loss, Trailing Stop, Time Exit)
        for t, info in list(portfolio.items()):
            row = full_test_data[(full_test_data.index == current_date) & (full_test_data['ticker'] == t)]
            if row.empty: continue

            market_price = float(row['Close'].iloc[0])
            high_price_today = float(row['High'].iloc[0]) if 'High' in row.columns else market_price

            # --- Update Trailing Stop ---
            info['highest_price_since_entry'] = max(info['highest_price_since_entry'], high_price_today)
            atr_exit = float(row['raw_atr'].iloc[0]) if 'raw_atr' in row.columns and not pd.isna(row['raw_atr'].iloc[0]) else market_price * 0.05
            if atr_exit > 0: # Ensure ATR is positive
                 potential_trailing_stop = info['highest_price_since_entry'] - (trailing_stop_atr_mult * atr_exit)
                 # Stop can only move up, never down
                 info['current_stop_loss_price'] = max(info['initial_stop_loss_price'], potential_trailing_stop)
            # --- End Trailing Stop Update ---

            exit_reason = None
            if market_price <= info['current_stop_loss_price']:
                exit_reason = 'stop-loss (initial or trailing)'
            elif (current_date - info['entry_date']).days >= holding_period:
                exit_reason = 'time-exit'

            if exit_reason:
                price_after_slippage = market_price * (1 - slip_pct)
                proceeds = info['shares'] * price_after_slippage
                transaction_cost = proceeds * cost_pct
                cash += (proceeds - transaction_cost)

                profit = (proceeds - transaction_cost) - info['cost_basis']
                trades.append({
                    'exit_date': current_date, 'ticker': t, 'profit': profit,
                    'entry_regime': info['entry_regime'], 'exit_reason': exit_reason,
                    'entry_date': info['entry_date'], 'entry_price': info['entry_price'],
                    'exit_price': market_price, 'shares': info['shares']
                })
                del portfolio[t]

        # 3. Process Entries
        todays_market_data = full_test_data[full_test_data.index == current_date]
        if todays_market_data.empty: continue

        # --- Check Regime Filter ---
        # Ensure predicted_regime column exists and handle potential NaN
        predicted_regime_val = todays_market_data['predicted_regime'].iloc[0] if 'predicted_regime' in todays_market_data.columns else np.nan
        if pd.isna(predicted_regime_val) or int(predicted_regime_val) not in allowed_entry_regimes:
            continue # Skip entries if regime is invalid or not allowed

        predicted_regime = int(predicted_regime_val) # Cast to int for storage

        # Get today's signals
        todays_preds = predictions_df[predictions_df['date'] == current_date].sort_values('pred_bk', ascending=False)

        for _, signal in todays_preds.iterrows():
            ticker = signal['ticker']
            # Apply confidence threshold
            if signal['pred_bk'] < confidence_thresh or ticker in portfolio:
                continue

            row = todays_market_data[todays_market_data['ticker'] == ticker]
            if not row.empty:
                price = float(row['Close'].iloc[0])
                atr = float(row['raw_atr'].iloc[0]) if 'raw_atr' in row.columns and not pd.isna(row['raw_atr'].iloc[0]) else price * 0.05
                if atr <= 0 or price <= 0: continue

                # --- ATR-Based Initial Stop ---
                stop_loss_distance = initial_stop_atr_mult * atr
                initial_stop_loss_price = price - stop_loss_distance
                if initial_stop_loss_price <= 0: continue

                # Position sizing based on ATR stop risk
                dollar_risk_per_share = price - initial_stop_loss_price
                if dollar_risk_per_share <= 0: continue

                portfolio_risk_amount = current_portfolio_value * risk_per_trade
                num_shares_to_buy = portfolio_risk_amount / dollar_risk_per_share
                amount_to_invest = num_shares_to_buy * price

                price_after_slippage = price * (1 + slip_pct)
                transaction_cost = amount_to_invest * cost_pct
                total_cost = amount_to_invest + transaction_cost

                if cash >= total_cost:
                    final_shares = amount_to_invest / price_after_slippage
                    cash -= total_cost

                    portfolio[ticker] = {
                        'shares': final_shares,
                        'entry_price': price_after_slippage,
                        'entry_date': current_date,
                        'entry_regime': predicted_regime,
                        'cost_basis': total_cost,
                        'initial_stop_loss_price': initial_stop_loss_price,
                        'current_stop_loss_price': initial_stop_loss_price,
                        'highest_price_since_entry': price # Use entry price initially
                    }

    return pd.DataFrame(portfolio_values).set_index('date').sort_index(), pd.DataFrame(trades)


# ---------------- Diagnostics & Analysis ----------------
# (Functions inspect_label_quality, compute_breakout_sample_weights,
#  analyze_performance_by_regime, save_results_summary are unchanged)
def inspect_label_quality(df, label_col='breakout_next_N', prob_col=None, save_prefix='diagnostic'):
    counts = df[label_col].value_counts()
    total = len(df)
    pos = counts.get(1,0); neg = counts.get(0,0)
    logging.info("Label counts -> total: %d, pos: %d (%.2f%%), neg: %d (%.2f%%)", total, pos, pos/total*100 if total else 0, neg, neg/total*100 if total else 0)

    if prob_col and prob_col in df.columns and len(df[label_col].unique()) > 1:
        from sklearn.metrics import roc_curve, precision_recall_curve, auc
        y_true = df[label_col].values; y_score = df[prob_col].values

        valid_idx = ~np.isnan(y_score)
        if np.sum(valid_idx) < len(y_true):
             logging.warning(f"Found {np.sum(~valid_idx)} NaN scores in predictions. Removing them for metric calculation.")
             y_true = y_true[valid_idx]
             y_score = y_score[valid_idx]

        if len(np.unique(y_true)) < 2:
            logging.warning(f"Only one class present in y_true after NaN removal. Skipping ROC/PR plots.")
            return {}

        fpr, tpr, _ = roc_curve(y_true, y_score); precision, recall, _ = precision_recall_curve(y_true, y_score)
        roc_auc = auc(fpr,tpr); pr_auc = auc(recall,precision)
        logging.info("Model Stats -> ROC AUC: %.4f | PR AUC: %.4f", roc_auc, pr_auc)
        try:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1); plt.plot(fpr,tpr); plt.title(f'ROC Curve (AUC={roc_auc:.3f})')
            plt.xlabel('FPR'); plt.ylabel('TPR')
            plt.subplot(1, 2, 2); plt.plot(recall,precision); plt.title(f'PR Curve (AUC={pr_auc:.3f})')
            plt.xlabel('Recall'); plt.ylabel('Precision')
            plt.tight_layout(); plt.savefig(f'{save_prefix}_roc_pr_curves.png'); plt.close()
            logging.info(f"Saved {save_prefix}_roc_pr_curves.png")
        except Exception as e:
            logging.warning(f"Could not save ROC/PR plot: {e}")
        return {'roc_auc': roc_auc, 'pr_auc': pr_auc}
    return {}

def compute_breakout_sample_weights(y):
    classes = np.unique(y)
    if len(classes) < 2:
        return np.ones_like(y, dtype=np.float32)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weight_map = {int(k): float(v) for k, v in zip(classes, weights)}
    sample_weights = np.array([class_weight_map.get(label, 1.0) for label in y], dtype=np.float32)
    return sample_weights


def analyze_performance_by_regime(trades_df: pd.DataFrame, regime_map: dict) -> dict:
    if trades_df.empty or 'entry_regime' not in trades_df.columns: return {}
    logging.info("--- Performance by Regime ---")
    regime_names = {k: v for k, v in regime_map.items()}
    trades_df['entry_regime'] = pd.to_numeric(trades_df['entry_regime'], errors='coerce')
    trades_df['regime_name'] = trades_df['entry_regime'].map(regime_names).fillna('Unknown')

    perf_stats = {}
    for name, group in trades_df.groupby('regime_name'):
        total_trades = len(group)
        if total_trades == 0: continue
        win_rate = (group['profit'] > 0).mean() * 100
        avg_profit = group['profit'].mean()
        total_profit = group['profit'].sum()
        stats = {
            'total_trades': total_trades,
            'win_rate_pct': win_rate,
            'average_profit': avg_profit,
            'total_profit': total_profit
        }
        perf_stats[name] = stats
        logging.info(f"Regime '{name}': {total_trades} trades, Win Rate: {win_rate:.2f}%, Total Profit: {total_profit:.2f}")
    return perf_stats

def save_results_summary(json_filename: str, strategy_params: dict, financial_metrics: dict, statistical_metrics: dict):
    # Ensure nested dicts/lists are serializable
    def make_serializable(obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        return obj

    summary = {
        'run_timestamp': datetime.now().isoformat(),
        'strategy_parameters': make_serializable(strategy_params),
        'financial_performance': make_serializable(financial_metrics),
        'model_statistics': make_serializable(statistical_metrics)
    }
    with open(json_filename, 'w') as f:
        json.dump(summary, f, indent=4)
    logging.info(f"Saved run summary to {json_filename}")

# ---------------- Main pipeline ----------------
def main(args):
    # --- Setup Logging ---
    run_id = args.run_id
    os.makedirs('backtests', exist_ok=True)

    LOG_FILENAME = LOG_FILENAME_TPL.format(run_id=run_id)
    RESULTS_FILENAME = RESULTS_FILENAME_TPL.format(run_id=run_id)
    TRADES_FILENAME = TRADES_FILENAME_TPL.format(run_id=run_id)
    PREDICTIONS_FILENAME = PREDICTIONS_FILENAME_TPL.format(run_id=run_id)
    EQUITY_CURVE_FILENAME = EQUITY_CURVE_FILENAME_TPL.format(run_id=run_id)
    JSON_SUMMARY_FILENAME = JSON_SUMMARY_FILENAME_TPL.format(run_id=run_id)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILENAME),
            logging.StreamHandler()
        ]
    )
    # --- End Logging Setup ---

    tf.random.set_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE); random.seed(RANDOM_STATE)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # --- Combine default params with CLI args ---
    strategy_params = DEFAULT_STRATEGY_PARAMS.copy()
    strategy_params.update({
        'confidence_threshold': args.confidence,
        'initial_stop_atr_multiplier': args.init_stop_atr,
        'trailing_stop_atr_multiplier': args.trail_stop_atr,
        'holding_period_days': args.hold_period,
        'risk_per_trade_percent': args.risk_per_trade,
        'allowed_entry_regimes': args.entry_regimes # Use CLI value
    })
    # Add back fixed params needed by process_data_for_ticker
    strategy_params['atr_multiplier'] = args.atr_label # For label definition
    # Store transaction costs separately if needed, remove from main dict if they cause issues elsewhere
    # strategy_params['transaction_cost_percent'] = DEFAULT_STRATEGY_PARAMS['transaction_cost_percent']
    # strategy_params['slippage_percent'] = DEFAULT_STRATEGY_PARAMS['slippage_percent']


    logging.info(f"--- STARTING RUN: {run_id} ---")
    logging.info(f"Logging to {LOG_FILENAME}")
    logging.info(f"Offline Mode: {args.offline}")
    logging.info(f"Strategy parameters: {strategy_params}")

    # --- Configure GPU Memory Growth ---
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"Enabled memory growth for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            logging.error(f"Could not set GPU memory growth: {e}")

    # --- Phase 0: Data Sync ---
    data_manager = DataManager(DB_CACHE_FILE, offline_mode=args.offline)
    base_tickers, ticker_to_sector = run_data_sync_phase(data_manager, offline_mode=args.offline)

    if not base_tickers or (not ticker_to_sector and not args.offline):
        logging.error("Data sync phase failed. Could not get tickers or sector map. Aborting.")
        return

    # --- Phase 1: Check Regime Model ---
    if not train_or_load_regime_model(data_manager): # Checks for files
        logging.error("Could not load regime model artifacts. Aborting.")
        return

    regime_model, regime_scaler, regime_features, regime_map = load_regime_model_and_artifacts()
    if not regime_model: return

    # --- Phase 2: Generate Regime Predictions (CHECKPOINTED) ---
    if os.path.exists(REGIME_PREDS_CACHE):
        logging.info(f"Loading cached regime predictions from {REGIME_PREDS_CACHE}")
        predicted_regimes_df = pd.read_pickle(REGIME_PREDS_CACHE)
    else:
        # (Prediction logic remains unchanged)
        if args.offline:
            logging.error(f"Offline mode: Regime predictions cache not found at {REGIME_PREDS_CACHE}. Aborting.")
            return

        logging.info("Phase 2: Generating and caching regime predictions...")
        regime_context_data = data_manager.fetch_all_data(REGIME_TICKERS)
        if not all(t in regime_context_data for t in REGIME_TICKERS):
             logging.error("Missing critical data for regime prediction. Aborting.")
             return

        market_df = regime_context_data["SPY"].copy()
        for t in REGIME_TICKERS:
            clean_name = t.replace('^','') + '_Close'
            if t != "SPY" and t in regime_context_data:
                 market_df = market_df.join(regime_context_data[t][['Close']].rename(columns={'Close': clean_name}))

        market_df['log_return'] = np.log(market_df['Close'] / market_df['Close'].shift(1))
        market_df['volatility_20d'] = market_df['log_return'].rolling(20).std() * np.sqrt(252)
        if 'TNX_Close' in market_df and 'FVX_Close' in market_df:
            market_df['yield_curve_5y10y'] = market_df['TNX_Close'] - market_df['FVX_Close']
        else: market_df['yield_curve_5y10y'] = np.nan
        if 'HYG_Close' in market_df and 'IEF_Close' in market_df:
            market_df['credit_spread_ratio'] = market_df['HYG_Close'] / market_df['IEF_Close']
        else: market_df['credit_spread_ratio'] = np.nan

        market_df.ffill(inplace=True); market_df.bfill(inplace=True)
        market_df = create_regime_features(market_df)
        market_df.fillna(0, inplace=True)

        predicted_regimes_df = predict_regimes_for_period(market_df, regime_model, regime_scaler, regime_features)
        if predicted_regimes_df.empty:
            logging.error("Failed to generate regime predictions. Aborting.")
            return

        predicted_regimes_df.to_pickle(REGIME_PREDS_CACHE)
        logging.info(f"Saved regime predictions to {REGIME_PREDS_CACHE}")

    # --- Phase 3: Load Breakout Data (CHECKPOINTED) ---
    global FEATURES # Declare FEATURES as global so process_data_for_ticker can use it
    if os.path.exists(BREAKOUT_DATASET_CACHE) and os.path.exists(BREAKOUT_FEATURES_CACHE):
        logging.info(f"Loading cached breakout dataset from {BREAKOUT_DATASET_CACHE}")
        full_dataset = pd.read_pickle(BREAKOUT_DATASET_CACHE)
        with open(BREAKOUT_FEATURES_CACHE, 'r') as f:
            FEATURES = json.load(f)
        logging.info(f"Loaded {len(FEATURES)} features from {BREAKOUT_FEATURES_CACHE}")
    else:
        # (Data processing logic remains unchanged)
        if args.offline:
            logging.error(f"Offline mode: Breakout dataset cache not found. Aborting.")
            return

        logging.info("Phase 3: Processing features for Breakout Model")
        sector_etfs = list(set(SECTOR_ETF_MAP.values()))
        data_map = data_manager.fetch_all_data(list(set(base_tickers + sector_etfs + ['SPY'])))

        context_dfs = {'predicted_regimes': predicted_regimes_df}
        if 'SPY' in data_map:
            spy_df = data_map['SPY'].copy()
            spy_df['SPY_Close'] = spy_df['Close']
            context_dfs['SPY'] = spy_df

        for etf in sector_etfs:
            if etf in data_map:
                etf_df = data_map[etf].copy()
                etf_df['Sector_Close'] = etf_df['Close']
                etf_df['sector_sma_50_rel'] = (etf_df['Sector_Close'] / etf_df['Sector_Close'].rolling(50).mean()) - 1.0
                context_dfs[etf] = etf_df

        processed = []
        logging.info(f"Processing features for {len(base_tickers)} tickers...")

        # Determine features dynamically first
        temp_df_for_cols = pd.DataFrame(columns=['Open','High','Low','Close','Volume']) # Dummy df
        temp_df_processed = process_data_for_ticker("TEMP", temp_df_for_cols, {}, {}, strategy_params)
        exclude = ['Open','High','Low','Close','Volume','Dividends','Stock Splits',
                   'ticker','SPY_Close','Sector_Close', 'raw_atr',
                   'future_return_N','breakout_next_N','Date', 'OBV']
        if temp_df_processed is not None:
             FEATURES = [c for c in temp_df_processed.columns if c not in exclude and 'Close' not in c]
        else:
             logging.warning("Could not automatically determine features. Using a default set.")
             FEATURES = ['SMA_50_rel', 'ROC_20', 'Volume_rel', 'RSI_14', 'ADX_14', 'CMF_20', 'STOCHk_14_3_3',
                         'historical_vol_20', 'ATR_14_rel', 'relative_strength_market', 'relative_strength_sector',
                         'month_sin', 'month_cos', 'predicted_regime']

        for t in base_tickers:
            if t in data_map:
                p = process_data_for_ticker(t, data_map[t], context_dfs, ticker_to_sector, strategy_params)
                if p is not None:
                    processed.append((t, p))

        valid_processed = [(t,df) for t,df in processed if df is not None]
        if not valid_processed:
            logging.error("No processed tickers; aborting.")
            return

        full_dataset = pd.concat([df.assign(ticker=t) for t,df in valid_processed]).sort_index()
        if full_dataset.empty:
            logging.error("Full dataset empty after concat.")
            return

        FEATURES = [c for c in full_dataset.columns if c not in exclude and 'Close' not in c]

        full_dataset.to_pickle(BREAKOUT_DATASET_CACHE)
        with open(BREAKOUT_FEATURES_CACHE, 'w') as f:
            json.dump(FEATURES, f)
        logging.info(f"Saved breakout dataset and features to cache.")

    logging.info(f"Using {len(FEATURES)} features for breakout model.")
    logging.info(f"Features are: {FEATURES}")

    full_dataset['Date'] = full_dataset.index
    inspect_label_quality(full_dataset, label_col='breakout_next_N', prob_col=None, save_prefix='full_dataset')
    logging.info("Data loaded. Deferring sequence creation to save memory.")


    # --- Phase 4: Tuner (AGGRESSIVE Memory-Optimized) ---
    # (Logic unchanged, uses tf.data.Dataset)
    if os.path.exists(BEST_HPS_CACHE) and not args.force_retune:
        logging.info(f"Loading cached best hyperparameters from {BEST_HPS_CACHE}")
        with open(BEST_HPS_CACHE, 'r') as f:
            best_hps = json.load(f)
    else:
        if args.offline:
            logging.error(f"Offline mode: Best HPs cache not found at {BEST_HPS_CACHE}. Cannot run tuner. Aborting.")
            return

        if args.force_retune:
            logging.info("Forcing re-tune as per --force-retune flag.")
        logging.info("Phase 4: Starting tuner search (AGGRESSIVE Memory-Optimized)...")

        logging.info("Creating sequences for tuner...")
        X_all_seq, y_bk_all_seq, y_ret_all_seq, _ = create_breakout_sequences(full_dataset, FEATURES, TIME_STEPS)
        if len(X_all_seq) == 0:
            logging.error("No sequences generated for tuner. Aborting.")
            return

        tuner_indices = list(range(int(len(X_all_seq) * 0.8)))
        X_tuner = X_all_seq[tuner_indices]
        y_bk_tuner = y_bk_all_seq[tuner_indices]
        y_ret_tuner = y_ret_all_seq[tuner_indices]

        logging.info("Freeing base sequence array from RAM...")
        del X_all_seq, y_bk_all_seq, y_ret_all_seq
        gc.collect()

        logging.info("Cleaning tuner data with np.nan_to_num...")
        np.nan_to_num(X_tuner, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        y_bk_tuner = np.nan_to_num(y_bk_tuner, nan=0.0)
        y_ret_tuner = np.nan_to_num(y_ret_tuner, nan=0.0, posinf=100.0, neginf=-50.0)

        logging.info("Scaling tuner data...")
        scaler_tuner = MinMaxScaler().fit(X_tuner.reshape(-1, X_tuner.shape[-1]))
        X_tuner_s = scaler_tuner.transform(X_tuner.reshape(-1, X_tuner.shape[-1])).reshape(X_tuner.shape)

        del X_tuner
        gc.collect()

        def build_for_tuner(hp): return build_model_from_hp(hp, len(FEATURES))
        tuner = kt.Hyperband(build_for_tuner, objective=kt.Objective("val_breakout_auc", direction="max"),
                             max_epochs=TUNER_EPOCHS, factor=3, directory='keras_tuner_dir', project_name='breakout_v5_combined', overwrite=True)

        logging.info("Starting tuner.search() with tf.data.Dataset...")
        split_idx = int(len(X_tuner_s) * 0.8)

        X_train_tuner = X_tuner_s[:split_idx]
        y_bk_train_tuner = y_bk_tuner[:split_idx]
        y_ret_train_tuner = y_ret_tuner[:split_idx]
        breakout_weights_tuner = compute_breakout_sample_weights(y_bk_train_tuner)
        train_dataset = create_tf_dataset(
            X_train_tuner, y_bk_train_tuner, y_ret_train_tuner,
            breakout_weights_tuner,
            TUNER_BATCH_SIZE, shuffle=True
        )

        X_val_tuner = X_tuner_s[split_idx:]
        y_bk_val_tuner = y_bk_tuner[split_idx:]
        y_ret_val_tuner = y_ret_tuner[split_idx:]
        val_weights_tuner = np.ones_like(y_bk_val_tuner, dtype=np.float32)
        val_dataset = create_tf_dataset(
            X_val_tuner, y_bk_val_tuner, y_ret_val_tuner,
            val_weights_tuner,
            TUNER_BATCH_SIZE
        )

        tuner.search(train_dataset,
                     validation_data=val_dataset,
                     epochs=TUNER_EPOCHS,
                     callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])

        del X_train_tuner, y_bk_train_tuner, y_ret_train_tuner, breakout_weights_tuner, train_dataset
        del X_val_tuner, y_bk_val_tuner, y_ret_val_tuner, val_weights_tuner, val_dataset
        del X_tuner_s, y_bk_tuner, y_ret_tuner
        gc.collect()

        best_hps_raw = tuner.get_best_hyperparameters(num_trials=1)[0].values
        best_hps = {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v) for k, v in best_hps_raw.items()}

        with open(BEST_HPS_CACHE, 'w') as f:
            json.dump(best_hps, f)
        logging.info(f"Saved best hyperparameters to {BEST_HPS_CACHE}")

    logging.info(f"Using hyperparameters: {best_hps}")

    # --- Phase 5: Purged K-Fold CV (RESUMABLE) ---
    # (Logic unchanged, uses tf.data.Dataset and yields weights)
    logging.info("Phase 5: Starting/Resuming Purged K-Fold Cross-Validation...")

    logging.info("Creating all sequences for K-Fold validation...")
    X_all_seq, y_bk_all_seq, y_ret_all_seq, seq_info_all = create_breakout_sequences(full_dataset, FEATURES, TIME_STEPS)
    if len(X_all_seq) == 0:
        logging.error("No sequences generated for K-Fold. Aborting.")
        return
    logging.info(f"Successfully created {len(X_all_seq)} sequences for K-Fold.")

    all_portfolio_dfs = []
    for fold_num in range(1, 6):
        pf_cache_file = PORTFOLIO_FOLD_CACHE_TPL.format(run_id=run_id, fold_id=fold_num)
        if os.path.exists(pf_cache_file):
            all_portfolio_dfs.append(pd.read_pickle(pf_cache_file))

    if all_portfolio_dfs:
        all_portfolio_dfs.sort(key=lambda df: df.index.min())
        logging.info(f"Loaded {len(all_portfolio_dfs)} completed portfolio folds from cache.")

    completed_folds = set()
    if os.path.exists(PREDICTIONS_FILENAME):
        try:
            existing_preds = pd.read_csv(PREDICTIONS_FILENAME)
            if 'fold' in existing_preds.columns:
                completed_folds = set(existing_preds['fold'].unique())
                logging.info(f"Found completed folds in {PREDICTIONS_FILENAME}: {completed_folds}")
        except pd.errors.EmptyDataError:
            pass

    t1_series = pd.Series(pd.to_datetime([info['date'] for info in seq_info_all]), index=range(len(seq_info_all)))
    t1_series += pd.Timedelta(days=BREAKOUT_PERIOD_DAYS)

    pkf = PurgedKFold(n_splits=5, t1=t1_series, pctEmbargo=0.01)

    num_sequences = len(seq_info_all)
    for fold, (train_idx, test_idx) in enumerate(pkf.split(np.arange(num_sequences))):
        fold_id = fold + 1

        if fold_id in completed_folds:
            logging.info(f"--- SKIPPING FOLD {fold_id}/5 (already completed) ---")
            continue

        logging.info(f"--- STARTING FOLD {fold_id}/5 ---")
        K.clear_session()

        X_tr, ybk_tr, yret_tr = X_all_seq[train_idx], y_bk_all_seq[train_idx], y_ret_all_seq[train_idx]
        X_ts, ybk_ts, yret_ts = X_all_seq[test_idx], y_bk_all_seq[test_idx], y_ret_all_seq[test_idx]
        seq_info_ts = [seq_info_all[i] for i in test_idx]

        np.nan_to_num(X_tr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(X_ts, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        ybk_tr = np.nan_to_num(ybk_tr, nan=0.0)
        ybk_ts = np.nan_to_num(ybk_ts, nan=0.0)
        yret_tr = np.nan_to_num(yret_tr, nan=0.0, posinf=100.0, neginf=-50.0)
        yret_ts = np.nan_to_num(yret_ts, nan=0.0, posinf=100.0, neginf=-50.0)

        if len(X_tr) == 0 or len(X_ts) == 0:
            logging.warning(f"Skipping fold {fold_id} due to insufficient sequences.")
            continue

        scaler = MinMaxScaler().fit(X_tr.reshape(-1, X_tr.shape[-1]))
        X_tr_s = scaler.transform(X_tr.reshape(-1, X_tr.shape[-1])).reshape(X_tr.shape)
        X_ts_s = scaler.transform(X_ts.reshape(-1, X_ts.shape[-1])).reshape(X_ts.shape)

        model = build_model_from_dict(best_hps, len(FEATURES))

        breakout_sample_weights_tr = compute_breakout_sample_weights(ybk_tr)

        train_dataset_fold = create_tf_dataset(
            X_tr_s, ybk_tr, yret_tr,
            breakout_sample_weights_tr,
            BATCH_SIZE, shuffle=True
        )

        logging.info(f"Fold {fold_id}: Starting model.fit for {EPOCHS_PER_FOLD} epochs...")

        model.fit(train_dataset_fold,
                  epochs=EPOCHS_PER_FOLD,
                  verbose=2, # Print one line per epoch
                  callbacks=[EarlyStopping(monitor='loss', patience=7, restore_best_weights=True)])

        logging.info(f"Fold {fold_id}: model.fit completed.")

        del train_dataset_fold, breakout_sample_weights_tr
        gc.collect()

        logging.info(f"Fold {fold_id}: Running predictions...")
        pred_bk, pred_ret = model.predict(X_ts_s)

        preds_df = pd.DataFrame(seq_info_ts)
        preds_df['pred_bk'] = pred_bk.ravel(); preds_df['pred_ret'] = pred_ret.ravel()
        preds_df['actual_bk'] = ybk_ts; preds_df['actual_ret'] = yret_ts
        preds_df['fold'] = fold_id

        test_dates = preds_df['date'].unique()
        test_df_fold = full_dataset[full_dataset['Date'].isin(test_dates)]

        initial_cash_fold = all_portfolio_dfs[-1]['total_value'].iloc[-1] if all_portfolio_dfs else INITIAL_PORTFOLIO_CASH

        pf_df, trades_df = run_backtest_on_fold(preds_df, test_df_fold, strategy_params, initial_cash_fold)

        if not pf_df.empty:
            all_portfolio_dfs.append(pf_df)
            pf_cache_file = PORTFOLIO_FOLD_CACHE_TPL.format(run_id=run_id, fold_id=fold_id)
            pf_df.to_pickle(pf_cache_file)
            logging.info(f"Saved portfolio for fold {fold_id} to {pf_cache_file}")

        if not trades_df.empty:
            trades_df['fold'] = fold_id
            trades_df.to_csv(TRADES_FILENAME, mode='a', header=not os.path.exists(TRADES_FILENAME), index=False)
            logging.info(f"Appended {len(trades_df)} trades for fold {fold_id} to {TRADES_FILENAME}")

        preds_df.to_csv(PREDICTIONS_FILENAME, mode='a', header=not os.path.exists(PREDICTIONS_FILENAME), index=False)
        logging.info(f"Appended {len(preds_df)} predictions for fold {fold_id} to {PREDICTIONS_FILENAME}")
        logging.info(f"--- COMPLETED FOLD {fold_id}/5 ---")

    # --- Phase 6: Final Evaluation and Saving ---
    # (Logic unchanged)
    logging.info("Phase 6: Final Evaluation and Model Saving")

    final_preds_df = pd.DataFrame()
    final_trades_df = pd.DataFrame()
    final_pf_df = pd.DataFrame()

    financial_metrics = {}
    statistical_metrics = {}

    if os.path.exists(PREDICTIONS_FILENAME):
        try:
            final_preds_df = pd.read_csv(PREDICTIONS_FILENAME).drop_duplicates()
            statistical_metrics = inspect_label_quality(final_preds_df, 'actual_bk', 'pred_bk', f'backtests/final_oos_{run_id}')
            if 'actual_ret' in final_preds_df.columns and 'pred_ret' in final_preds_df.columns:
                 mae = mean_absolute_error(final_preds_df['actual_ret'], final_preds_df['pred_ret'])
                 rmse = (mean_squared_error(final_preds_df['actual_ret'], final_preds_df['pred_ret']))**0.5
                 statistical_metrics.update({'mae': mae, 'rmse': rmse})
                 logging.info(f"Final OOS Return Stats -> MAE: {mae:.4f} | RMSE: {rmse:.4f}")
            else:
                 logging.warning("Return columns missing from predictions file, skipping MAE/RMSE.")
        except Exception as e:
            logging.error(f"Could not load or analyze {PREDICTIONS_FILENAME}: {e}")

    if os.path.exists(TRADES_FILENAME):
        try:
            final_trades_df = pd.read_csv(TRADES_FILENAME).drop_duplicates()
        except Exception as e:
            logging.error(f"Could not load {TRADES_FILENAME}: {e}")

    if all_portfolio_dfs:
        final_pf_df = pd.concat(all_portfolio_dfs).sort_index()
        final_pf_df = final_pf_df[~final_pf_df.index.duplicated(keep='first')]

        pf_series = final_pf_df['total_value']
        if not pf_series.empty:
            final_val = float(pf_series.iloc[-1])
            total_return = (final_val / INITIAL_PORTFOLIO_CASH - 1.0) * 100
            peak = pf_series.cummax(); dd = (pf_series - peak) / peak; max_dd = float(dd.min()*100)

            daily_returns = pf_series.pct_change().fillna(0)
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0

            total_trades = len(final_trades_df)
            win_count = sum(final_trades_df['profit'] > 0) if not final_trades_df.empty else 0
            win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0.0

            financial_metrics = {
                'total_return_pct': total_return,
                'max_drawdown_pct': max_dd,
                'sharpe_ratio': sharpe,
                'total_trades': total_trades,
                'win_rate_pct': win_rate
            }

            regime_performance = analyze_performance_by_regime(final_trades_df, regime_map if 'regime_map' in locals() else REGIME_NAME_MAP)
            financial_metrics['performance_by_regime'] = regime_performance

            logging.info("--- Final Financial Performance ---")
            logging.info(f"Total Return: {total_return:.2f}%")
            logging.info(f"Max Drawdown: {max_dd:.2f}%")
            logging.info(f"Sharpe Ratio: {sharpe:.3f}")
            logging.info(f"Win Rate: {win_rate:.2f}% ({total_trades} trades)")

            try:
                plt.figure(figsize=(15,7)); plt.plot(pf_series.index, pf_series.values)
                plt.title(f'Combined Equity Curve v5 (Run: {args.run_id})'); plt.ylabel('Portfolio Value'); plt.grid(True)
                plt.savefig(EQUITY_CURVE_FILENAME); plt.close()
                logging.info(f"Saved equity curve to {EQUITY_CURVE_FILENAME}")
            except Exception as e:
                logging.warning(f"Could not save equity curve plot: {e}")
        else:
             logging.warning("Portfolio series is empty. Skipping financial calculations.")
    else:
        logging.warning("No portfolio dataframes were found or generated. Skipping financial analysis.")


    with open(RESULTS_FILENAME, 'w') as f:
        f.write(f"Run ID: {args.run_id} | Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Offline Mode: {args.offline}\n")
        f.write("="*80 + "\n")
        f.write(f"Best Hyperparameters: {best_hps}\n")
        f.write(f"Strategy Parameters: {strategy_params}\n")
        f.write("="*80 + "\n")
        f.write("MODEL STATISTICS (Out-of-Sample)\n")
        for k, v in statistical_metrics.items(): f.write(f"  {k}: {v:.4f}\n")
        f.write("="*80 + "\n")
        f.write("FINANCIAL PERFORMANCE (Out-of-Sample)\n")
        for k, v in financial_metrics.items():
            if k != 'performance_by_regime': f.write(f"  {k}: {v}\n")
        f.write("="*80 + "\n")

    save_results_summary(JSON_SUMMARY_FILENAME, strategy_params, financial_metrics, statistical_metrics)

    # --- Phase 7: Retrain Final Breakout Model on All Data ---
    # (Logic unchanged, uses tf.data.Dataset and yields weights)
    if args.save_model:
        if args.offline:
            logging.warning("Offline mode: Skipping final model training (--save-model ignored).")
        else:
            logging.info("Phase 7: Retraining final breakout model on all data...")

            logging.info("Cleaning final training data...")
            np.nan_to_num(X_all_seq, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            y_bk_all_seq = np.nan_to_num(y_bk_all_seq, nan=0.0)
            y_ret_all_seq = np.nan_to_num(y_ret_all_seq, nan=0.0, posinf=100.0, neginf=-50.0)

            final_scaler = MinMaxScaler().fit(X_all_seq.reshape(-1, X_all_seq.shape[-1]))
            X_all_s = final_scaler.transform(X_all_seq.reshape(-1, X_all_seq.shape[-1])).reshape(X_all_seq.shape)
            final_model = build_model_from_dict(best_hps, len(FEATURES))

            breakout_sample_weights_final = compute_breakout_sample_weights(y_bk_all_seq)

            final_train_dataset = create_tf_dataset(
                X_all_s, y_bk_all_seq, y_ret_all_seq,
                breakout_sample_weights_final,
                BATCH_SIZE, shuffle=True
            )

            final_model.fit(final_train_dataset,
                            epochs=EPOCHS_PER_FOLD, verbose=2)

            logging.info("Saving final breakout model and artifacts.")
            final_model.save(BREAKOUT_MODEL_FILENAME)
            with open(BREAKOUT_SCALER_FILENAME, 'wb') as f: pickle.dump(final_scaler, f)
            metadata = {
                'features': FEATURES,
                'best_hps': best_hps,
                'strategy_params': strategy_params,
                'time_steps': TIME_STEPS
            }
            with open(BREAKOUT_METADATA_FILENAME, 'wb') as f: pickle.dump(metadata, f)
    else:
        logging.info("Skipping final model save as --save-model was not specified.")

    logging.info(f"--- RUN {run_id} COMPLETE ---")


def setup_arg_parser():
    parser = argparse.ArgumentParser(description="Gated Walk-Forward Breakout Model Trainer (v6 Improved Strategy).")
    parser.add_argument('--run-id', type=str, default=f"run_{datetime.now().strftime('%Y%m%d_%H%M')}", help="Unique ID for this run. Re-use to resume.")

    # Strategy Parameters
    parser.add_argument('--confidence', type=float, default=DEFAULT_STRATEGY_PARAMS['confidence_threshold'], help="Model confidence threshold for entry.")
    parser.add_argument('--init-stop-atr', type=float, default=DEFAULT_STRATEGY_PARAMS['initial_stop_atr_multiplier'], help="ATR multiplier for initial stop loss distance.")
    parser.add_argument('--trail-stop-atr', type=float, default=DEFAULT_STRATEGY_PARAMS['trailing_stop_atr_multiplier'], help="ATR multiplier for trailing stop loss distance.")
    parser.add_argument('--hold-period', type=int, default=DEFAULT_STRATEGY_PARAMS['holding_period_days'], help="Maximum holding period in days.")
    parser.add_argument('--risk-per-trade', type=float, default=DEFAULT_STRATEGY_PARAMS['risk_per_trade_percent'], help="Portfolio risk % per trade for sizing.")
    parser.add_argument('--entry-regimes', type=int, nargs='+', default=DEFAULT_STRATEGY_PARAMS['allowed_entry_regimes'], help="List of regime numbers allowed for entry (e.g., 0 1).")
    parser.add_argument('--atr-label', type=float, default=3.0, help="ATR multiplier used ONLY for defining the breakout label during data processing.") # Kept separate for label stability

    # Model Saving & Checkpointing
    parser.add_argument('--save-model', action='store_true', help="If set, retrain and save the final breakout model artifacts.")
    parser.add_argument('--force-retune', action='store_true', help="Force Keras Tuner to re-run, ignoring cached hyperparameters.")
    parser.add_argument('--offline', action='store_true', help="Run in offline mode. Fails if any data is missing from cache.")

    return parser

if __name__ == "__main__":
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()
    main(args)