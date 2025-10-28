#!/usr/bin/env python3
# breakout-trainer-v5-final.py
"""
Breakout Trainer v5 - Final Version
- Imports tickers from tickers.json
- Purged and Embargoed K-Fold Cross-Validation for robust backtesting
- Advanced Feature Engineering (OBV, CMF, Stochastics, Volatility)
- Deeper CNN architecture with Layer Normalization
- Dynamic learning rate scheduling and Huber loss for robust training
"""

import os
import logging
import pickle
import random
import time
import json # <-- Import json library
from datetime import datetime
from io import StringIO
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import pandas_ta as ta
import tensorflow as tf
import keras_tuner as kt
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, mean_squared_error

from tensorflow.keras import Input, Model, optimizers, backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv1D, Dense, Dropout, LSTM, GRU, MaxPooling1D, Bidirectional, GlobalAveragePooling1D
from tensorflow.keras import layers

# ---------------- Config ----------------
HISTORY_PERIOD_YEARS = 5
MIN_HISTORY_REQUIREMENT = 252 * 3
TIME_STEPS = 60
BREAKOUT_PERIOD_DAYS = 15
RANDOM_STATE = 42
BATCH_SIZE = 32
CACHE_DIR = "data_cache"
TICKERS_JSON_FILE = "news_analysis.json" # <-- Filename for the custom tickers

RESULTS_FILENAME = "backtest_results_v5_final.txt"
TRADES_FILENAME = "backtest_trades_v5_final.csv"
PREDICTIONS_FILENAME = "oos_predictions_v5_final.csv"
MODEL_FILENAME = "final_tuned_model_v5_final.keras"
SCALER_FILENAME = "final_scaler_v5_final.pkl"
METADATA_FILENAME = "final_metadata_v5_final.pkl"

TUNER_EPOCHS = 15
MAX_TRIALS = 35
EPOCHS_PER_FOLD = 15

INITIAL_PORTFOLIO_CASH = 100000.0
POSITION_SIZE_PERCENT = 0.05
STRATEGY_PARAMS = {
    'confidence_threshold': 0.65,
    'stop_loss_percent': 0.06,
    'holding_period_days': 15,
    'atr_multiplier': 3.0
}

SECTOR_ETF_MAP = {
    'Technology': 'XLK', 'Financials': 'XLF', 'Health Care': 'XLV',
    'Consumer Discretionary': 'XLY', 'Industrials': 'XLI', 'Consumer Staples': 'XLP',
    'Energy': 'XLE', 'Utilities': 'XLU', 'Real Estate': 'XLRE',
    'Materials': 'XLB', 'Communication Services': 'XLC'
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------- Helpers: Data ----------------
def get_tickers_from_json(filename: str) -> list:
    """Loads tickers from a local JSON file."""
    if not os.path.exists(filename):
        logging.warning(f"'{filename}' not found. Skipping custom tickers.")
        return []
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        tickers = list(data.keys())
        logging.info(f"Loaded {len(tickers)} tickers from '{filename}'.")
        return tickers
    except Exception as e:
        logging.error(f"Could not read or parse '{filename}': {e}")
        return []

def get_sp500_tickers() -> list:
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=10); r.raise_for_status()
        table = pd.read_html(StringIO(r.text))
        tickers = table[0]['Symbol'].tolist()
        #return [t.replace('.', '-') for t in tickers]
        return []
    except Exception as e:
        logging.error("Could not scrape S&P500 tickers: %s", e)
        return []

def get_stock_data_yf(ticker: str) -> Tuple[str, pd.DataFrame]:
    cache_path = os.path.join(CACHE_DIR, f"{ticker}.pkl")
    if os.path.exists(cache_path):
        try:
            return ticker, pd.read_pickle(cache_path)
        except Exception:
            pass
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=f"{HISTORY_PERIOD_YEARS}y", auto_adjust=True)
        if df.empty: return ticker, None
        df.index = df.index.tz_localize(None)
        df.to_pickle(cache_path)
        return ticker, df.rename_axis('Date')
    except Exception as e:
        logging.warning("yfinance fetch failed for %s: %s", ticker, e)
        return ticker, None

def fetch_all_tickers_yf(tickers: List[str], max_workers: int = 12) -> List[Tuple[str, pd.DataFrame]]:
    logging.info("Fetching data for %d tickers from yfinance...", len(tickers))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(get_stock_data_yf, tickers))
    valid = [(t, df) for t, df in results if df is not None]
    logging.info("Fetched valid data for %d/%d tickers.", len(valid), len(tickers))
    return valid

def get_vix_data_cboe() -> pd.DataFrame:
    cache = os.path.join(CACHE_DIR, "^VIX.pkl")
    if os.path.exists(cache):
        try:
            return pd.read_pickle(cache)
        except Exception:
            pass
    try:
        url = 'https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv'
        df = pd.read_csv(url)
        df['Date'] = pd.to_datetime(df['DATE']); df.set_index('Date', inplace=True)
        df.index = df.index.tz_localize(None)
        df.rename(columns={'CLOSE':'Close','OPEN':'Open','HIGH':'High','LOW':'Low'}, inplace=True)
        df['Volume'] = 0
        df = df[['Open','High','Low','Close','Volume']]
        df.to_pickle(cache)
        logging.info("Fetched & cached VIX data.")
        return df
    except Exception as e:
        logging.error("Could not fetch VIX: %s", e)
        return None

# ---------------- Feature engineering (log-return regression) ----------------
def process_data_for_ticker(ticker: str, df: pd.DataFrame, context_dfs: dict, ticker_to_sector: dict) -> pd.DataFrame:
    if df is None or len(df) < MIN_HISTORY_REQUIREMENT: return None
    df = df.copy().sort_index()
    if 'SPY' in context_dfs:
        spy_context_cols = ['SPY_Close', 'market_regime']
        df = df.merge(context_dfs['SPY'][spy_context_cols], left_index=True, right_index=True, how='left')
    if '^VIX' in context_dfs:
        df = df.merge(context_dfs['^VIX'][['VIX_SMA_20_rel','VIX_ROC_5']], left_index=True, right_index=True, how='left')
    sector = ticker_to_sector.get(ticker)
    if sector and sector in SECTOR_ETF_MAP:
        etf = SECTOR_ETF_MAP[sector]
        secdf = context_dfs.get(etf)
        if secdf is not None:
            df = df.merge(secdf[['Sector_Close','sector_sma_50_rel']], left_index=True, right_index=True, how='left')
    df.ffill(inplace=True)

    close = df['Close']
    df['SMA_50_rel'] = (close / close.rolling(50).mean()) - 1.0
    df['ROC_20'] = close.pct_change(20)
    df['Volume_rel'] = (df['Volume'] / df['Volume'].rolling(20).mean()) - 1.0
    try:
        df.ta.rsi(length=14, append=True)
        df.ta.adx(length=14, append=True)
        df.ta.obv(append=True)
        df.ta.cmf(length=20, append=True)
        stoch = df.ta.stoch(k=14, d=3, smooth_k=3)
        if isinstance(stoch, pd.DataFrame):
            df['STOCHk_14_3_3'] = stoch['STOCHk_14_3_3']
            df['STOCHd_14_3_3'] = stoch['STOCHd_14_3_3']
        
        log_return = np.log(df['Close'] / df['Close'].shift(1))
        df['historical_vol_20'] = log_return.rolling(window=20).std() * np.sqrt(252)
        df['garman_klass_vol'] = ((np.log(df['High']) - np.log(df['Low']))**2 / 2 - (2 * np.log(2) - 1) * (np.log(df['Close']) - np.log(df['Open']))**2)
        raw_atr = df.ta.atr(length=14)
    except Exception:
        raw_atr = close.rolling(14).apply(lambda x: (x.max()-x.min()) if len(x)>0 else np.nan)

    df['ATR_14_rel'] = raw_atr / close
    df['relative_strength_market'] = df['Close'] / df.get('SPY_Close', close)
    if 'Sector_Close' in df.columns:
        df['relative_strength_sector'] = df['Close'] / df['Sector_Close']

    idx = df.index
    df['month_sin'] = np.sin(2*np.pi*idx.month/12)
    df['month_cos'] = np.cos(2*np.pi*idx.month/12)
    df['day_of_week_sin'] = np.sin(2*np.pi*idx.dayofweek/7)
    df['day_of_week_cos'] = np.cos(2*np.pi*idx.dayofweek/7)

    future_price = close.shift(-BREAKOUT_PERIOD_DAYS)
    log_ret = np.log1p((future_price / close) - 1.0)
    df['future_return_N'] = log_ret * 100.0

    future_highs = df['High'].shift(-1).rolling(window=BREAKOUT_PERIOD_DAYS).max()
    breakout_level = df['Close'] + (raw_atr * STRATEGY_PARAMS.get('atr_multiplier', 3.0))
    df['breakout_next_N'] = (future_highs > breakout_level).astype(int)

    df.dropna(inplace=True)
    if len(df) < MIN_HISTORY_REQUIREMENT: return None
    return df

# ---------------- Purged K-Fold CV ----------------
class PurgedKFold(KFold):
    def __init__(self, n_splits=10, t1=None, pctEmbargo=0.01):
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pctEmbargo = pctEmbargo

    def split(self, X, y=None, groups=None):
        if X.empty:
            raise ValueError("X cannot be empty")
        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0] * self.pctEmbargo)
        test_starts = [(i[0], i[-1] + 1) for i in np.array_split(indices, self.n_splits)]
        for i, j in test_starts:
            t0 = self.t1[i] if self.t1 is not None else X.index[i]
            test_indices = indices[i:j]
            t1 = self.t1[j - 1] if self.t1 is not None else X.index[j - 1]
            
            # Embargo
            train_indices = np.concatenate([
                indices[:i],
                indices[j + mbrg:]
            ])
            
            # Purge
            if self.t1 is not None:
                train_t1 = self.t1[train_indices]
                train_indices = train_indices[np.where(train_t1 < t0)[0]]
                train_indices = train_indices[np.where(train_t1 > t1)[0]]
            
            yield train_indices, test_indices

# ---------------- Sequence creation ----------------
def create_sequences_for_fold(df_fold: pd.DataFrame, features: List[str], time_steps: int):
    X,y_bk,y_ret,seq_info = [],[],[],[]
    for ticker, group in df_fold.groupby('ticker'):
        data = group[features].values
        bk = group['breakout_next_N'].values
        ret = group['future_return_N'].values.clip(-50, 100)
        n = len(group)
        if n <= time_steps: continue
        for i in range(time_steps, n):
            X.append(data[i-time_steps:i])
            y_bk.append(bk[i])
            y_ret.append(ret[i])
            seq_info.append({'ticker': ticker, 'date': group.index[i]})
    return np.array(X), np.array(y_bk), np.array(y_ret), seq_info

# ---------------- Model builders (automatic TF RNN fallback) ----------------
def build_model_from_hp(hp, n_features):
    conv_filters_1 = hp.Int('conv_filters_1', 32, 128, step=32, default=64)
    conv_filters_2 = hp.Int('conv_filters_2', 32, 64, step=16, default=48)
    kernel_size = hp.Choice('kernel_size', [3, 5, 7], default=5)
    rnn_type = hp.Choice('rnn_type', ['lstm', 'gru'], default='lstm')
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
    x = layers.LayerNormalization()(x)

    if rnn_type == 'lstm':
        rnn_layer = LSTM(units=rnn_units, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout)
    else:
        rnn_layer = GRU(units=rnn_units, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout)

    x = Bidirectional(rnn_layer)(x)
    x = layers.LayerNormalization()(x)

    if use_attention:
        score = layers.Dense(1, activation='tanh')(x)
        score = layers.Softmax(axis=1)(score)
        x = layers.Multiply()([x, score])
        x = layers.GlobalAveragePooling1D()(x)
    else:
        x = GlobalAveragePooling1D()(x)

    x = Dropout(dropout)(x)
    shared = Dense(shared_units, activation='relu')(x)
    shared = layers.LayerNormalization()(shared)

    bk = Dense(head_units, activation='relu')(shared)
    bk = Dropout(dropout)(bk)
    breakout_output = Dense(1, activation='sigmoid', name='breakout')(bk)

    ret = Dense(head_units, activation='relu')(shared)
    ret = Dropout(dropout)(ret)
    return_output = Dense(1, activation='linear', name='return')(ret)

    model = Model(inputs=inp, outputs=[breakout_output, return_output])
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss={'breakout': 'binary_crossentropy', 'return': tf.keras.losses.Huber()},
                  loss_weights={'breakout': 1.0, 'return': 0.5},
                  metrics={'breakout': [tf.keras.metrics.AUC(name='auc')], 'return': ['mae']})
    return model

def build_model_from_dict(hp_values: dict, n_features: int):
    conv_filters_1 = int(hp_values.get('conv_filters_1', 64))
    conv_filters_2 = int(hp_values.get('conv_filters_2', 48))
    kernel_size = int(hp_values.get('kernel_size', 5))
    rnn_type = hp_values.get('rnn_type', 'lstm')
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
    x = layers.LayerNormalization()(x)

    if rnn_type == 'lstm':
        rnn_layer = LSTM(units=rnn_units, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout)
    else:
        rnn_layer = GRU(units=rnn_units, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout)

    x = Bidirectional(rnn_layer)(x)
    x = layers.LayerNormalization()(x)

    if use_attention:
        score = layers.Dense(1, activation='tanh')(x)
        score = layers.Softmax(axis=1)(score)
        x = layers.Multiply()([x, score])
        x = layers.GlobalAveragePooling1D()(x)
    else:
        x = GlobalAveragePooling1D()(x)

    x = Dropout(dropout)(x)
    shared = Dense(shared_units, activation='relu')(x)
    shared = layers.LayerNormalization()(shared)

    bk = Dense(head_units, activation='relu')(shared)
    bk = Dropout(dropout)(bk)
    breakout_output = Dense(1, activation='sigmoid', name='breakout')(bk)

    ret = Dense(head_units, activation='relu')(shared)
    ret = Dropout(dropout)(ret)
    return_output = Dense(1, activation='linear', name='return')(ret)

    model = Model(inputs=inp, outputs=[breakout_output, return_output])
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss={'breakout': 'binary_crossentropy', 'return': tf.keras.losses.Huber()},
                  loss_weights={'breakout': 1.0, 'return': 0.5},
                  metrics={'breakout': [tf.keras.metrics.AUC(name='auc')], 'return': ['mae']})
    return model

# ---------------- Backtester (robust accounting) ----------------
def run_backtest_on_fold(predictions_df: pd.DataFrame, full_test_data: pd.DataFrame, strategy_params: dict, initial_cash: float):
    logging.info("Running backtest from %s to %s", predictions_df['date'].min(), predictions_df['date'].max())
    cash = float(initial_cash)
    portfolio = {}
    trades = []
    portfolio_values = []

    unique_dates = sorted(pd.to_datetime(predictions_df['date'].unique()))
    for current_date in unique_dates:
        current_portfolio_value = cash
        for t, info in list(portfolio.items()):
            row = full_test_data[(full_test_data.index == current_date) & (full_test_data['ticker'] == t)]
            market_price = float(row['Close'].iloc[0]) if not row.empty else info['entry_price']
            current_portfolio_value += info['shares'] * market_price

        portfolio_values.append({'date': current_date, 'total_value': current_portfolio_value})

        for t, info in list(portfolio.items()):
            row = full_test_data[(full_test_data.index == current_date) & (full_test_data['ticker'] == t)]
            market_price = float(row['Close'].iloc[0]) if not row.empty else info['entry_price']
            stop_price = info['entry_price'] * (1 - strategy_params['stop_loss_percent'])
            exit_reason = None
            if market_price <= stop_price:
                exit_reason = 'stop-loss'
            elif (current_date - info['entry_date']).days >= strategy_params['holding_period_days']:
                exit_reason = 'time-exit'
            if exit_reason:
                cash += info['shares'] * market_price
                profit = (market_price - info['entry_price']) * info['shares']
                trades.append({'date': current_date, 'ticker': t, 'action': f'sell ({exit_reason})', 'price': market_price, 'shares': info['shares'], 'profit': profit})
                del portfolio[t]

        todays_preds = predictions_df[predictions_df['date'] == current_date].sort_values('pred_bk', ascending=False)
        available_capital = current_portfolio_value
        for _, signal in todays_preds.iterrows():
            if signal['pred_bk'] <= strategy_params['confidence_threshold']:
                continue
            t = signal['ticker']
            if t in portfolio: continue
            row = full_test_data[(full_test_data.index == current_date) & (full_test_data['ticker'] == t)]
            if row.empty: continue
            price = float(row['Close'].iloc[0])
            amount = available_capital * POSITION_SIZE_PERCENT
            if cash >= amount and amount > 0:
                shares = amount / price
                cash -= amount
                portfolio[t] = {'shares': shares, 'entry_price': price, 'entry_date': current_date}
                trades.append({'date': current_date, 'ticker': t, 'action': 'buy', 'price': price, 'shares': shares})

    pf_df = pd.DataFrame(portfolio_values).set_index('date').sort_index()
    return pf_df, trades

# ---------------- Diagnostics & imbalance helpers ----------------
def inspect_label_quality(df, label_col='breakout_next_N', prob_col=None, save_prefix='diagnostic'):
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    counts = df[label_col].value_counts()
    total = len(df)
    pos = counts.get(1,0); neg = counts.get(0,0)
    logging.info("Label counts -> total: %d, pos: %d (%.2f%%), neg: %d (%.2f%%)", total, pos, pos/total*100 if total else 0, neg, neg/total*100 if total else 0)
    plt.figure(figsize=(5,3)); plt.bar([0,1],[neg,pos], tick_label=['no','yes']); plt.title('Label distribution'); plt.tight_layout(); plt.savefig(f'{save_prefix}_label_hist.png'); plt.close()
    logging.info("Saved %s_label_hist.png", save_prefix)
    if prob_col and prob_col in df.columns:
        y_true = df[label_col].values; y_score = df[prob_col].values
        fpr, tpr, _ = roc_curve(y_true, y_score); precision, recall, _ = precision_recall_curve(y_true, y_score)
        logging.info("ROC AUC: %.4f | PR AUC: %.4f", auc(fpr,tpr), auc(recall,precision))
        plt.figure(); plt.plot(fpr,tpr); plt.title('ROC'); plt.savefig(f'{save_prefix}_roc.png'); plt.close()
        plt.figure(); plt.plot(recall,precision); plt.title('PR'); plt.savefig(f'{save_prefix}_pr.png'); plt.close()

def compute_breakout_sample_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return np.array([dict(zip(classes, weights)).get(label,1.0) for label in y])

# ---------------- Main pipeline ----------------
def main():
    tf.random.set_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE); random.seed(RANDOM_STATE)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs('backtests', exist_ok=True)
    if os.path.exists(TRADES_FILENAME): os.remove(TRADES_FILENAME)
    if os.path.exists(PREDICTIONS_FILENAME): os.remove(PREDICTIONS_FILENAME)

    logging.info("Phase 1: Data collection & features (yfinance)")
    vix = get_vix_data_cboe()
    if vix is None: return
    context = {'^VIX': vix}

    # --- Updated Ticker Loading ---
    tickers_sp = get_sp500_tickers()
    custom_tickers = get_tickers_from_json(TICKERS_JSON_FILE)
    
    base = sorted(list(set(tickers_sp + custom_tickers)))
    logging.info("Found a total of %d unique tickers to process.", len(base))

    ticker_to_sector = {}
    for t in base:
        try:
            info = yf.Ticker(t).info
            sec = info.get('sector')
            if sec: ticker_to_sector[t] = sec
        except Exception:
            continue

    valid_tickers = [t for t,s in ticker_to_sector.items() if s in SECTOR_ETF_MAP]
    if not valid_tickers:
        valid_tickers = base[:50]

    sector_etfs = list(set(SECTOR_ETF_MAP.values()))
    to_fetch = sorted(list(set(valid_tickers + sector_etfs + ['SPY'])))
    data = fetch_all_tickers_yf(to_fetch)
    data_map = {t:df for t,df in data}

    if 'SPY' in data_map:
        spy_df = data_map['SPY']
        spy_df['SPY_Close'] = spy_df['Close']
        spy_ma_50 = spy_df['Close'].rolling(window=50).mean()
        spy_ma_200 = spy_df['Close'].rolling(window=200).mean()
        spy_df['market_regime'] = (spy_ma_50 > spy_ma_200).astype(int)
    
    context.update(data_map)
    context['^VIX']['VIX_SMA_20_rel'] = (context['^VIX']['Close'] / context['^VIX']['Close'].rolling(20).mean()) - 1.0
    context['^VIX']['VIX_ROC_5'] = context['^VIX']['Close'].pct_change(5)
    for etf in sector_etfs:
        if etf in context:
            context[etf]['Sector_Close'] = context[etf]['Close']
            context[etf]['sector_sma_50_rel'] = (context[etf]['Sector_Close'] / context[etf]['Sector_Close'].rolling(50).mean()) - 1.0

    ticker_dfs = [(t, data_map[t]) for t in valid_tickers if t in data_map]
    processed = []
    for t, df in ticker_dfs:
        p = process_data_for_ticker(t, df, context, ticker_to_sector)
        processed.append((t,p))

    valid_processed = [(t,df) for t,df in processed if df is not None]
    if not valid_processed:
        logging.error("No processed tickers; aborting.")
        return

    full_dataset = pd.concat([df.assign(ticker=t) for t,df in valid_processed]).sort_index()
    if full_dataset.empty:
        logging.error("Full dataset empty after concat.")
        return

    exclude = ['Open','High','Low','Close','Volume','Dividends','Stock Splits','ticker','SPY_Close','Sector_Close','future_return_N','breakout_next_N','Date', 'OBV']
    FEATURES = [c for c in full_dataset.columns if c not in exclude]
    logging.info("Using %d features.", len(FEATURES))
    logging.info(f"Features are: {FEATURES}")

    full_dataset['Date'] = full_dataset.index.tz_localize(None)
    inspect_label_quality(full_dataset, label_col='breakout_next_N', prob_col=None, save_prefix='full_dataset')
    
    X_all_seq, y_bk_all_seq, y_ret_all_seq, seq_info_all = create_sequences_for_fold(full_dataset, FEATURES, TIME_STEPS)

    if len(X_all_seq) == 0:
        logging.error("No sequences generated from the full dataset. Aborting.")
        return
    
    # --- Tuner using a single split ---
    logging.info("Starting tuner search...")
    train_indices = list(range(int(len(X_all_seq) * 0.8)))
    val_indices = list(range(int(len(X_all_seq) * 0.8), len(X_all_seq)))
    
    X_train, y_bk_train, y_ret_train = X_all_seq[train_indices], y_bk_all_seq[train_indices], y_ret_all_seq[train_indices]
    X_val, y_bk_val, y_ret_val = X_all_seq[val_indices], y_bk_all_seq[val_indices], y_ret_all_seq[val_indices]

    scaler_tuner = MinMaxScaler().fit(X_train.reshape(-1, X_train.shape[-1]))
    X_train_s = scaler_tuner.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_s = scaler_tuner.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

    def build_for_tuner(hp): return build_model_from_hp(hp, len(FEATURES))
    tuner = kt.Hyperband(build_for_tuner, objective=kt.Objective("val_breakout_auc", direction="max"),
                         max_epochs=TUNER_EPOCHS, factor=3, directory='keras_tuner_dir', project_name='breakout_v5_final', overwrite=True)
    
    tuner_callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    ]
    tuner.search(X_train_s, [y_bk_train, y_ret_train], validation_data=(X_val_s, [y_bk_val, y_ret_val]),
                 epochs=TUNER_EPOCHS, batch_size=BATCH_SIZE, callbacks=tuner_callbacks)
    
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0].values
    logging.info("Best hyperparameters: %s", best_hps)

    # --- Purged K-Fold Cross-Validation ---
    all_portfolio = []
    all_trades = []
    
    t1 = pd.Series(pd.to_datetime([info['date'] for info in seq_info_all]), index=range(len(seq_info_all)))
    pkf = PurgedKFold(n_splits=5, t1=t1, pctEmbargo=0.01)

    for fold, (train_idx, test_idx) in enumerate(pkf.split(pd.DataFrame(index=t1.index))):
        logging.info(f"--- FOLD {fold+1}/5 ---")
        K.clear_session()

        X_tr, ybk_tr, yret_tr = X_all_seq[train_idx], y_bk_all_seq[train_idx], y_ret_all_seq[train_idx]
        X_ts, ybk_ts, yret_ts = X_all_seq[test_idx], y_bk_all_seq[test_idx], y_ret_all_seq[test_idx]
        seq_info_ts = [seq_info_all[i] for i in test_idx]

        if len(X_tr) == 0 or len(X_ts) == 0:
            logging.warning("Skipping fold due to insufficient sequences.")
            continue
        
        scaler = MinMaxScaler().fit(X_tr.reshape(-1, X_tr.shape[-1]))
        X_tr_s = scaler.transform(X_tr.reshape(-1, X_tr.shape[-1])).reshape(X_tr.shape)
        X_ts_s = scaler.transform(X_ts.reshape(-1, X_ts.shape[-1])).reshape(X_ts.shape)

        breakout_sample_weights = compute_breakout_sample_weights(ybk_tr)
        model = build_model_from_dict(best_hps, len(FEATURES))
        
        fold_callbacks = [
            EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=1e-6)
        ]
        model.fit(X_tr_s, [ybk_tr, yret_tr], epochs=EPOCHS_PER_FOLD, batch_size=BATCH_SIZE, verbose=0,
                  sample_weight=[breakout_sample_weights, np.ones_like(yret_tr, dtype=float)], callbacks=fold_callbacks)

        pred_bk, pred_ret = model.predict(X_ts_s)
        preds_df = pd.DataFrame(seq_info_ts)
        preds_df['pred_bk'] = pred_bk.ravel(); preds_df['pred_ret'] = pred_ret.ravel()
        preds_df['actual_bk'] = ybk_ts; preds_df['actual_ret'] = yret_ts
        
        test_dates = preds_df['date'].unique()
        test_df_fold = full_dataset[full_dataset['Date'].isin(test_dates)]
        
        initial_cash = all_portfolio[-1]['total_value'].iloc[-1] if all_portfolio else INITIAL_PORTFOLIO_CASH
        pf, trades = run_backtest_on_fold(preds_df, test_df_fold, STRATEGY_PARAMS, initial_cash)
        
        if not pf.empty:
            all_portfolio.append(pf); all_trades.extend(trades)
            pd.DataFrame(trades).to_csv(TRADES_FILENAME, mode='a', header=not os.path.exists(TRADES_FILENAME), index=False)
            preds_df.to_csv(PREDICTIONS_FILENAME, mode='a', header=not os.path.exists(PREDICTIONS_FILENAME), index=False)

    # --- Final Evaluation and Saving ---
    with open(RESULTS_FILENAME, 'a') as f:
        f.write(f"Run Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Best Hyperparameters: {best_hps}\n")
        f.write(f"Strategy Parameters: {STRATEGY_PARAMS}\n")

        if os.path.exists(PREDICTIONS_FILENAME):
            final_preds = pd.read_csv(PREDICTIONS_FILENAME)
            try:
                auc_bk = roc_auc_score(final_preds['actual_bk'], final_preds['pred_bk'])
                ap_bk = average_precision_score(final_preds['actual_bk'], final_preds['pred_bk'])
                mae = mean_absolute_error(final_preds['actual_ret'], final_preds['pred_ret'])
                rmse = (mean_squared_error(final_preds['actual_ret'], final_preds['pred_ret']))**0.5
                f.write(f"STATISTICAL (Breakout) -> AUC: {auc_bk:.4f} | PR-AUC: {ap_bk:.4f}\n")
                f.write(f"STATISTICAL (Return) -> MAE: {mae:.4f} | RMSE: {rmse:.4f}\n")
            except Exception as e:
                logging.warning("Could not compute final stats: %s", e)

        if all_portfolio:
            final_pf = pd.concat(all_portfolio).sort_index()
            pf_series = final_pf['total_value'] if 'total_value' in final_pf.columns else final_pf.iloc[:,0]
            final_val = float(pf_series.iloc[-1])
            total_return = (final_val / INITIAL_PORTFOLIO_CASH - 1.0) * 100
            peak = pf_series.cummax(); dd = (pf_series - peak) / peak; max_dd = float(dd.min()*100)
            total_trades = len([t for t in all_trades if t.get('action','').startswith('sell')])
            win_count = sum(1 for t in all_trades if t.get('action','').startswith('sell') and t.get('profit',0) > 0)
            win_rate = (win_count / total_trades * 100) if total_trades>0 else 0.0
            f.write(f"FINANCIAL -> Total Return: {total_return:.2f}% | Max Drawdown: {max_dd:.2f}% | Win Rate: {win_rate:.2f}% ({total_trades} trades)\n")

            plt.figure(figsize=(12,6)); plt.plot(pf_series.index, pf_series.values); plt.title('Equity Curve v5 Final'); plt.ylabel('Portfolio Value'); plt.grid(True)
            plt.savefig('backtests/FINAL_equity_curve_v5_final.png'); plt.close()

        f.write("="*80 + "\n\n")

    logging.info("Retraining final model on all data...")
    final_scaler = MinMaxScaler().fit(X_all_seq.reshape(-1, X_all_seq.shape[-1]))
    X_all_s = final_scaler.transform(X_all_seq.reshape(-1, X_all_seq.shape[-1])).reshape(X_all_seq.shape)
    final_model = build_model_from_dict(best_hps, len(FEATURES))
    bw = compute_breakout_sample_weights(y_bk_all_seq)
    final_model.fit(X_all_s, [y_bk_all_seq, y_ret_all_seq], epochs=EPOCHS_PER_FOLD, batch_size=BATCH_SIZE, verbose=2,
                    sample_weight=[bw, np.ones_like(y_ret_all_seq, dtype=float)])

    logging.info("Saving final model and artifacts.")
    final_model.save(MODEL_FILENAME)
    with open(SCALER_FILENAME, 'wb') as f: pickle.dump(final_scaler, f)
    metadata = {'features': FEATURES, 'used_tickers': valid_tickers, 'best_hps': best_hps, 'strategy_params': STRATEGY_PARAMS}
    with open(METADATA_FILENAME, 'wb') as f: pickle.dump(metadata, f)

if __name__ == "__main__":
    main()