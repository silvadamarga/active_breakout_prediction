#!/usr/bin/env python3
"""
v6 - Configuration File
Stores global parameters and file paths.
Includes ENHANCED_FEATURES list (20 features).
Uses refined strategy parameters based on analysis.
"""

import os
from datetime import datetime

# ---------------- Basic Setup ----------------
HISTORY_PERIOD_YEARS = 15
MIN_HISTORY_REQUIREMENT = 252 * 3 # Min years of data for a stock to be included
TIME_STEPS = 60 # Sequence length for Keras model
BREAKOUT_PERIOD_DAYS = 30 # Increased for longer horizon prediction
RANDOM_STATE = 42

# ---------------- Data & Caching ----------------
CACHE_DIR = "data_cache_v6"
DB_CACHE_FILE = os.path.join(CACHE_DIR, "market_data_cache_v6.db")
TICKERS_JSON_FILE = "tickers.json" # Source for custom tickers

# --- Checkpoint Cache Filenames ---
SP500_TICKERS_CACHE = os.path.join(CACHE_DIR, "sp500_tickers.json")
SECTOR_MAP_CACHE = os.path.join(CACHE_DIR, "ticker_to_sector.json")
REGIME_PREDS_CACHE = os.path.join(CACHE_DIR, "checkpoint_v6_predicted_regimes.pkl")
ALL_FEATURES_CACHE = os.path.join(CACHE_DIR, "checkpoint_v6_all_features_enhanced.pkl") # Updated name
EVENT_MANIFEST_CACHE = os.path.join(CACHE_DIR, "checkpoint_v6_event_manifest_enhanced.pkl") # Updated name
BEST_HPS_CACHE = os.path.join(CACHE_DIR, "checkpoint_v6_best_hps_enhanced.json") # Updated name
COMBINED_PF_CACHE_TPL = os.path.join(CACHE_DIR, "checkpoint_v6_combined_pf_{run_id}.pkl")


# ---------------- Regime Model Artifacts ----------------
REGIME_MODEL_FILENAME = "market_regime_model_v4.keras"
REGIME_SCALER_FILENAME = "market_regime_scaler_v4.pkl"
REGIME_METADATA_FILENAME = "market_regime_metadata_v4.pkl"
REGIME_TICKERS = ["SPY", "^VIX", "^TNX", "^FVX", "HYG", "IEF", "GLD"]
# Set bullish regimes based on latest analysis (Moderate and High Risk)
BULLISH_REGIMES = [0, 1, 2, 3]
REGIME_FUTURE_PERIOD_DAYS = 5


# ---------------- Breakout Model Training ----------------
# Update artifact names to reflect enhanced features
BREAKOUT_MODEL_FILENAME = "final_tuned_model_v7_enhanced.txt"
BREAKOUT_SCALER_FILENAME = "final_scaler_v7_enhanced.pkl"
BREAKOUT_METADATA_FILENAME = "final_metadata_v7_enhanced.pkl"

OPTUNA_N_TRIALS = 100
OPTUNA_TIMEOUT = 5 + 60 + 60

TUNER_EPOCHS = 15
MAX_TRIALS = 35 # For Keras Tuner (Hyperband)
EPOCHS_PER_FOLD = 40 # Epochs for each K-Fold training
TUNER_BATCH_SIZE = 16
BATCH_SIZE = 16 # Batch size for K-Fold training and final model
N_SPLITS = 5 # Number of folds for Purged K-Fold CV

# Original Blueprint Features (14 total)
BLUEPRINT_FEATURES = [
    'BBW_norm', 'BBW_6mo_rank', 'ATR_norm', 'ATR_6mo_rank', 'Choppiness_14d',
    'ADX_14d', 'SMA_50_slope', 'OBV_slope_20d', 'Volume_vs_Avg',
    'RS_vs_SPX_slope', 'RS_vs_Sector_slope', 'VIX_level', 'VIX_MA_ratio',
    'predicted_regime'
]

# --- ENHANCED FEATURE LIST (Now 20 features total) ---
ENHANCED_FEATURES = BLUEPRINT_FEATURES.copy()
ENHANCED_FEATURES.extend([
    'RS_vs_Bonds_slope', # Added
    'pct_above_50d',    # Added
    'pct_above_200d',   # Added
    'SMA_20_50_Ratio',  # Added
    'ATR_norm_std_20d', # Added
    'ROC_20d'           # Added
])
# Optional: Add debug print to verify length when script runs
# print(f"DEBUG: Length of ENHANCED_FEATURES = {len(ENHANCED_FEATURES)}")
# print(f"DEBUG: ENHANCED_FEATURES = {sorted(ENHANCED_FEATURES)}") # Print sorted list


# ---------------- Backtest Strategy Config ----------------
INITIAL_PORTFOLIO_CASH = 10000.0 # Standard initial capital
DEFAULT_STRATEGY_PARAMS = {
    'confidence_threshold': 0.35, # Probability threshold for 'Win' class
    'holding_period_days': BREAKOUT_PERIOD_DAYS, # Use the updated longer horizon
    'transaction_cost_percent': 0.0005, # 0.05% per trade
    'slippage_percent': 0.0005, # 0.05% slippage on entry/exit
    'risk_per_trade_percent': 0.01, # Risk 1% (RESET TO 1% - CRITICAL FOR RISK MGMT)
    'use_regime_filter': True, # Default to ON
    # --- Trailing Stop ---
    'use_trailing_stop': True, # Default to OFF based on analysis
    'atr_stop_multiplier': 3.0, # Looser default if trailing stop enabled later
    # --- Fixed Stop (Fallback if trailing is off) ---
    'stop_loss_percent': 0.06, # Fixed 6% stop (used if use_trailing_stop=False)
}


# ---------------- File Paths for Output ----------------
RESULTS_FILENAME_TPL = "backtests/backtest_results_{run_id}.txt"
TRADES_FILENAME_TPL = "backtests/backtest_trades_{run_id}.csv"
PREDICTIONS_FILENAME_TPL = "backtests/oos_predictions_{run_id}.csv"
EQUITY_CURVE_FILENAME_TPL = "backtests/equity_curve_{run_id}.png"
JSON_SUMMARY_FILENAME_TPL = "backtests/summary_{run_id}.json"
LOG_FILENAME_TPL = "backtests/run_log_{run_id}.log"
PORTFOLIO_FOLD_CACHE_TPL = "backtests/pf_fold_{run_id}_{fold_id}.pkl"


# ---------------- Other Config ----------------
SECTOR_ETF_MAP = {
    'Technology': 'XLK', 'Financials': 'XLF', 'Health Care': 'XLV',
    'Consumer Discretionary': 'XLY', 'Industrials': 'XLI', 'Consumer Staples': 'XLP',
    'Energy': 'XLE', 'Utilities': 'XLU', 'Real Estate': 'XLRE',
    'Materials': 'XLB', 'Communication Services': 'XLC'
}

# Ensure directories exist
os.makedirs('backtests', exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)