#!/usr/bin/env python3
"""
v6 - Main Orchestrator Script
Runs the entire pipeline: Data -> Regime -> Breakout -> Backtest -> Save.
Uses ENHANCED_FEATURES (20 features) and LightGBM with Optuna tuning.
"""

import os
import logging
import pickle
import random
import time
import json
import argparse
import gc
from datetime import datetime
from typing import List, Dict, Tuple, Any

# Suppress excessive logging from libraries
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Still useful if TF is imported by dependencies
logging.getLogger('matplotlib').setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import pandas_ta as ta
# --- LightGBM & Optuna Imports ---
import lightgbm as lgb
import optuna
from sklearn.metrics import roc_auc_score # For Optuna objective
from sklearn.model_selection import train_test_split # For Optuna validation split
import joblib # For saving GBDT model

# --- Keep Necessary Sklearn Imports ---
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

# --- REMOVED Keras/TF Imports ---
# import tensorflow as tf # Keep if using Keras regime model
# import keras_tuner as kt
# from tensorflow.keras.callbacks import EarlyStopping # Only EarlyStopping from callbacks was used
# from tensorflow.keras.utils import to_categorical # No longer needed
# from tensorflow.keras import backend as K # No longer needed

# --- Project Module Imports ---

# Explicitly import needed config vars
from .config import (
    DEFAULT_STRATEGY_PARAMS, BULLISH_REGIMES, DB_CACHE_FILE, RANDOM_STATE,
    ALL_FEATURES_CACHE, EVENT_MANIFEST_CACHE, BEST_HPS_CACHE,
    ENHANCED_FEATURES, TIME_STEPS, BREAKOUT_PERIOD_DAYS, N_SPLITS,
    INITIAL_PORTFOLIO_CASH, # BATCH_SIZE might not be needed, EPOCHS_PER_FOLD used conceptually
    PREDICTIONS_FILENAME_TPL, TRADES_FILENAME_TPL, COMBINED_PF_CACHE_TPL,
    EQUITY_CURVE_FILENAME_TPL, JSON_SUMMARY_FILENAME_TPL, LOG_FILENAME_TPL,
    BREAKOUT_MODEL_FILENAME, BREAKOUT_SCALER_FILENAME, BREAKOUT_METADATA_FILENAME,
    CACHE_DIR, SECTOR_ETF_MAP,
    OPTUNA_N_TRIALS, OPTUNA_TIMEOUT # Import Optuna params
)

from .data_manager import (
    DataManager, get_sp500_tickers_cached,
    run_data_sync_phase
)
# Still need regime model loading which uses Keras/TF internally
from .regime_model import (
    train_or_load_regime_model, load_regime_model_and_artifacts,
    build_and_predict_regime_df
)
# --- Remove Keras Model Import ---
# from .breakout_model import build_model_from_hp, build_model_from_dict
from .validation import PurgedKFold
from .backtester import run_backtest_on_fold
from .utils import (
    # configure_gpu_memory_growth, # Not needed for CPU LGBM
    inspect_label_quality, compute_breakout_sample_weights,
    analyze_performance_by_regime, save_results_summary,
    calculate_financial_metrics, plot_equity_curve
)

# ---------------- Logging Setup ----------------
def setup_logging(run_id: str):
    """Initializes logging to file (append) and console."""
    log_filename = LOG_FILENAME_TPL.format(run_id=run_id)
    try: os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    except OSError as e: print(f"Error creating log directory: {e}")
    root_logger = logging.getLogger();
    # Close existing handlers associated with the root logger
    for handler in root_logger.handlers[:]: root_logger.removeHandler(handler); handler.close()
    log_handlers = [logging.StreamHandler()];
    try: log_handlers.append(logging.FileHandler(log_filename, mode='a'))
    except Exception as e: print(f"Error setting up file logger: {e}. Console only.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-5.5s] %(message)s", handlers=log_handlers)
    logging.getLogger('optuna').setLevel(logging.WARNING) # Reduce Optuna verbosity
    logging.getLogger('lightgbm').setLevel(logging.WARNING) # Reduce LightGBM verbosity
    # Suppress TF logs if it's still imported for the regime model
    try:
        import tensorflow as tf
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
    except ImportError:
        pass


# --- Optuna Objective Function ---
def objective(trial: optuna.Trial, X_train, y_train, X_val, y_val, w_train, w_val):
    """Objective function for Optuna hyperparameter tuning with LightGBM."""

    # Define hyperparameters to tune
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss', # Logloss is common for LGBM multiclass
        'num_class': 3, # Our 3 classes (Loss, Timeout, Win mapped to 0, 1, 2)
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000, step=100), # Wider range for n_estimators
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.15, log=True), # Slightly wider range
        'num_leaves': trial.suggest_int('num_leaves', 20, 200), # Wider range
        'max_depth': trial.suggest_int('max_depth', 3, 15), # Wider range
        'subsample': trial.suggest_float('subsample', 0.5, 1.0, step=0.05), # Finer steps
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.05), # Finer steps
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 15.0, log=True), # Wider L1
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 15.0, log=True), # Wider L2
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'seed': RANDOM_STATE,
        'n_jobs': -1, # Use all available CPU cores
        'verbose': -1, # Suppress LightGBM's own verbose output during tuning
        # 'class_weight': 'balanced' # Option 1: Let LGBM handle imbalance (often less effective than sample_weight)
        # Option 2: Use sample weights (passed as w_train, w_val)
    }

    # Initialize and train LightGBM model
    model = lgb.LGBMClassifier(**params)

    # Use sample_weight for imbalance handling
    model.fit(
        X_train, y_train,
        sample_weight=w_train, # Pass training sample weights
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[w_val], # Pass validation sample weights
        eval_metric='auc_mu', # Evaluate using macro-averaged multi-class AUC (robust to imbalance)
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)] # Early stopping based on validation AUC
    )

    # Predict probabilities on validation set
    y_pred_proba = model.predict_proba(X_val)

    # Calculate multi-class AUC (OvR Macro) - our primary metric
    try:
        # Ensure y_val has at least 2 classes for AUC calculation
        if len(np.unique(y_val)) < 2:
             logging.warning("Optuna trial: Validation set has only one class. Returning 0.5 AUC.")
             return 0.5 # Cannot calculate AUC, return neutral score
        # Use macro average AUC as the objective
        auc_score = roc_auc_score(y_val, y_pred_proba, multi_class='ovr', average='macro')
    except ValueError as e:
        logging.warning(f"Optuna trial: Could not calculate AUC: {e}. Returning 0.5.")
        auc_score = 0.5 # Return neutral score if AUC fails

    # Handle potential NaN/Inf scores
    return auc_score if np.isfinite(auc_score) else 0.5


# ---------------- Main Orchestrator ----------------
def main(args):
    start_time = time.time()
    run_id = args.run_id
    setup_logging(run_id)
    logging.info(f"--- STARTING GBDT RUN: {run_id} ---")
    logging.info(f"Timestamp: {datetime.now().isoformat()}")
    logging.info(f"Using feature set: ENHANCED_FEATURES ({len(ENHANCED_FEATURES)} features)")
    logging.info(f"Prediction Horizon (BREAKOUT_PERIOD_DAYS): {BREAKOUT_PERIOD_DAYS}")
    # ... (Log other args) ...

    np.random.seed(RANDOM_STATE); random.seed(RANDOM_STATE) # Set seeds

    # --- Strategy Params ---
    strategy_params = DEFAULT_STRATEGY_PARAMS.copy()
    # (Parameter update logic remains the same)
    arg_dict = vars(args); params_to_update = {}
    arg_map = {'confidence': 'confidence_threshold', 'risk_per_trade': 'risk_per_trade_percent',
               'hold_period': 'holding_period_days', 'atr_stop_mult': 'atr_stop_multiplier',
               'stop_loss': 'stop_loss_percent'}
    for arg_name, param_name in arg_map.items():
        if arg_dict.get(arg_name) is not None: params_to_update[param_name] = arg_dict[arg_name]
    params_to_update['use_regime_filter'] = args.regime_filter
    params_to_update['use_trailing_stop'] = args.use_trailing_stop
    params_to_update['BULLISH_REGIMES'] = BULLISH_REGIMES
    strategy_params.update(params_to_update)
    logging.info(f"Strategy parameters: {json.dumps(strategy_params, indent=2)}")

    # --- Phase 0: Data Sync ---
    data_manager = DataManager(DB_CACHE_FILE, offline_mode=args.offline)
    base_tickers, ticker_to_sector = run_data_sync_phase(data_manager, offline_mode=args.offline)
    if not base_tickers: logging.error("Data sync failed."); return 1

    # --- Phase 1 & 2: Regime Model & Predictions ---
    # (Unchanged - still use the Keras regime model)
    regime_model, regime_scaler, regime_features, regime_map = load_regime_model_and_artifacts()
    if not regime_model: logging.error("Regime model loading failed."); return 1
    predicted_regimes_df = build_and_predict_regime_df(
        data_manager, regime_model, regime_scaler, regime_features, args.offline
    )
    if predicted_regimes_df is None or predicted_regimes_df.empty:
        logging.error("Regime prediction failed."); return 1

    # --- Phase 3: Build Feature Database & Event Manifest ---
    # (Unchanged from previous refactored version - calculates enhanced features)
    all_features_df, event_manifest_df = pd.DataFrame(), pd.DataFrame()
    feature_cache_path = ALL_FEATURES_CACHE
    manifest_cache_path = EVENT_MANIFEST_CACHE
    # ... [ Cache loading / rebuilding logic remains the same ] ...
    if os.path.exists(feature_cache_path) and os.path.exists(manifest_cache_path) and not args.force_rebuild_features:
        logging.info(f"Loading cached features from {feature_cache_path}")
        try: all_features_df = pd.read_pickle(feature_cache_path)
        except Exception as e: logging.error(f"Error loading feature cache: {e}. Forcing rebuild."); args.force_rebuild_features = True
        logging.info(f"Loading cached manifest from {manifest_cache_path}")
        try: event_manifest_df = pd.read_pickle(manifest_cache_path)
        except Exception as e: logging.error(f"Error loading manifest cache: {e}. Forcing rebuild."); args.force_rebuild_features = True
    else:
        if args.offline: logging.error(f"Offline mode: Cache(s) not found/invalid."); return 1
        logging.info("Phase 3: Building Feature Database & Event Manifest...")
        # ... [ Breadth calculation and Context preparation logic ] ...
        logging.info("Calculating market breadth...")
        sp500_tickers_for_breadth = get_sp500_tickers_cached(offline_mode=args.offline)
        breadth_data = {}
        if sp500_tickers_for_breadth:
            sp500_data_map = data_manager.fetch_all_data(sp500_tickers_for_breadth)
            above_50d_list, above_200d_list = [], []
            spy_df_ref = data_manager.fetch_all_data(['SPY']).get('SPY')
            if spy_df_ref is not None:
                common_index = spy_df_ref.index.sort_values()
                processed_breadth_tickers = 0
                for ticker in sp500_tickers_for_breadth:
                    df = sp500_data_map.get(ticker)
                    if df is not None and not df.empty and 'Close' in df.columns:
                        try:
                            df_aligned = df.reindex(common_index).ffill()
                            if df_aligned['Close'].notna().sum() > 50:
                                sma_50 = ta.sma(df_aligned['Close'], 50)
                                above_50d_list.append((df_aligned['Close'] > sma_50) & df_aligned['Close'].notna() & sma_50.notna())
                            if df_aligned['Close'].notna().sum() > 200:
                                sma_200 = ta.sma(df_aligned['Close'], 200)
                                above_200d_list.append((df_aligned['Close'] > sma_200) & df_aligned['Close'].notna() & sma_200.notna())
                            processed_breadth_tickers += 1
                        except Exception: continue
                logging.info(f"Processed breadth calcs for {processed_breadth_tickers}/{len(sp500_tickers_for_breadth)} tickers.")
                if above_50d_list: breadth_data['pct_above_50d'] = pd.concat(above_50d_list, axis=1).astype(float).mean(axis=1) * 100
                if above_200d_list: breadth_data['pct_above_200d'] = pd.concat(above_200d_list, axis=1).astype(float).mean(axis=1) * 100
                if breadth_data: logging.info("Calculated breadth features.")
                else: logging.warning("Could not calculate breadth.")
            else: logging.warning("SPY needed for breadth index.")
        else: logging.warning("S&P500 list needed for breadth.")
        breadth_df = pd.DataFrame(breadth_data)
        # Context prep
        sector_etfs = list(set(SECTOR_ETF_MAP.values())); context_tickers = list(set(['SPY', '^VIX', 'TLT'] + sector_etfs))
        context_data_map = data_manager.fetch_all_data(context_tickers); context_dfs = {'predicted_regimes': predicted_regimes_df}
        if not breadth_df.empty: context_dfs['breadth'] = breadth_df
        def process_context_ticker(ticker_name, col_name, cols_to_keep, dfs_dict, data_map):
             if ticker_name in data_map:
                df = data_map[ticker_name].copy(); rename_dict = {'Close': col_name} if len(cols_to_keep) == 1 and cols_to_keep[0] == col_name else {}
                if rename_dict: df.rename(columns=rename_dict, inplace=True)
                if ticker_name == '^VIX':
                     if col_name in df.columns:
                         df['VIX_SMA_5'] = ta.sma(df[col_name], 5); df['VIX_SMA_20'] = ta.sma(df[col_name], 20)
                         cols_to_keep.extend(['VIX_SMA_5', 'VIX_SMA_20'])
                     else: logging.warning(f"Close missing for {ticker_name}")
                existing_cols = [c for c in cols_to_keep if c in df.columns]
                if existing_cols: dfs_dict[ticker_name.replace('^','')] = df[existing_cols]
                else: logging.warning(f"Cols {cols_to_keep} not found for {ticker_name}.")
             else: logging.warning(f"Data for {ticker_name} not found.")
        process_context_ticker('SPY', 'SPY_Close', ['SPY_Close'], context_dfs, context_data_map)
        process_context_ticker('^VIX', 'VIX_Close', ['VIX_Close'], context_dfs, context_data_map)
        process_context_ticker('TLT', 'TLT_Close', ['TLT_Close'], context_dfs, context_data_map)
        for etf in sector_etfs: process_context_ticker(etf, 'Sector_Close', ['Sector_Close'], context_dfs, context_data_map)
        # Build
        try:
            from .breakout_processor import build_event_manifest
            all_features_df, event_manifest_df = build_event_manifest(data_manager, base_tickers, context_dfs, ticker_to_sector)
        except Exception as e: logging.error(f"Feature build failed: {e}", exc_info=True); return 1
        if all_features_df.empty or event_manifest_df.empty: logging.error("Feature build empty."); return 1
        # Save
        try:
            logging.info(f"Saving features ({len(all_features_df)}) to {feature_cache_path}")
            all_features_df.to_pickle(feature_cache_path)
            logging.info(f"Saving manifest ({len(event_manifest_df)}) to {manifest_cache_path}")
            event_manifest_df.to_pickle(manifest_cache_path)
        except Exception as e: logging.error(f"Failed cache save: {e}")

    # --- End Feature Rebuild ---

    logging.info(f"Total events found: {len(event_manifest_df)}")
    if len(event_manifest_df) == 0: logging.error("No events available."); return 1
    # (Label distribution logging)
    try:
        label_counts = event_manifest_df['label'].value_counts().sort_index()
        label_map = {-1: 'Loss', 0: 'Timeout', 1: 'Win'}
        for label_val, count in label_counts.items():
            logging.info(f"  Label {label_map.get(label_val, f'({label_val})')}: {count} ({count/len(event_manifest_df)*100:.2f}%)")
    except Exception as e: logging.warning(f"Could not log label distribution: {e}")


    # --- Phase 4: Optuna Tuning for LightGBM ---
    best_hps = {}
    hps_cache_path = BEST_HPS_CACHE
    hps_cache_exists = os.path.exists(hps_cache_path)
    if hps_cache_exists and not args.force_retune:
        logging.info(f"Loading cached HPs from {hps_cache_path}")
        try:
            with open(hps_cache_path, 'r') as f: best_hps = json.load(f)
        except Exception as e: logging.error(f"Failed HPs cache load: {e}. Forcing re-tune."); args.force_retune = True

    if not best_hps or args.force_retune:
        if args.offline: logging.error(f"Offline mode: HPs cache needed."); return 1
        logging.info("Phase 4: Starting Optuna search for LightGBM with ENHANCED_FEATURES...")
        try: from .breakout_processor import create_breakout_sequences
        except ImportError: logging.error("Missing sequence fns."); return 1

        tuner_sample_size = min(len(event_manifest_df), 15000) # Keep sample size reasonable
        tuner_manifest = event_manifest_df.sample(n=tuner_sample_size, random_state=RANDOM_STATE)

        X_tuner_seq, y_tuner_seq, _ = create_breakout_sequences(tuner_manifest, all_features_df, ENHANCED_FEATURES, TIME_STEPS)
        if len(X_tuner_seq) == 0: logging.error("No sequences for tuner."); return 1
        logging.info(f"Generated {len(X_tuner_seq)} sequences for tuner.")

        # GBDT Data Prep
        n_samples, n_steps, n_features = X_tuner_seq.shape
        X_tuner_flat = X_tuner_seq.reshape(n_samples, n_steps * n_features)
        y_tuner = y_tuner_seq
        np.nan_to_num(X_tuner_flat, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        y_tuner = np.nan_to_num(y_tuner, nan=1)

        logging.info("Scaling tuner data (flattened)...")
        scaler_tuner = MinMaxScaler().fit(X_tuner_flat)
        X_tuner_s = scaler_tuner.transform(X_tuner_flat)
        del X_tuner_seq, X_tuner_flat; gc.collect()

        X_tr_opt, X_val_opt, y_tr_opt, y_val_opt = train_test_split(
            X_tuner_s, y_tuner, test_size=0.2, random_state=RANDOM_STATE, stratify=y_tuner
        )
        w_tr_opt = compute_breakout_sample_weights(y_tr_opt)
        w_val_opt = compute_breakout_sample_weights(y_val_opt)

        logging.info(f"Starting Optuna optimization ({OPTUNA_N_TRIALS} trials)...")
        # Ensure Optuna doesn't log too verbosely
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize", study_name=f"lgbm_{run_id}")

        try:
            study.optimize(
                lambda trial: objective(trial, X_tr_opt, y_tr_opt, X_val_opt, y_val_opt, w_tr_opt, w_val_opt),
                n_trials=OPTUNA_N_TRIALS,
                timeout=OPTUNA_TIMEOUT if OPTUNA_TIMEOUT is not None else None,
                gc_after_trial=True, show_progress_bar=True
            )
            best_hps = study.best_params
            logging.info(f"Optuna finished. Best Validation AUC: {study.best_value:.5f}")
            try:
                with open(hps_cache_path, 'w') as f: json.dump(best_hps, f, indent=4)
                logging.info(f"Saved best HPs to {hps_cache_path}")
            except Exception as e: logging.error(f"Failed HPs cache save: {e}")
        except Exception as optuna_err:
             logging.error(f"Error during Optuna search: {optuna_err}", exc_info=True)
             return 1
        finally:
             del X_tr_opt, y_tr_opt, X_val_opt, y_val_opt, w_tr_opt, w_val_opt
             del X_tuner_s, y_tuner, scaler_tuner; gc.collect()

    # --- End Optuna ---
    if not best_hps: logging.error("No valid HPs found/loaded."); return 1
    logging.info(f"Using best hyperparameters: {json.dumps(best_hps, indent=2)}")

    # --- Phase 5: Purged K-Fold CV with LightGBM ---
    logging.info("Phase 5: Starting Purged K-Fold CV with LightGBM...")
    try: from .breakout_processor import create_breakout_sequences
    except ImportError: logging.error("Missing sequence fns."); return 1

    logging.info("Creating all sequences for K-Fold...")
    X_all_seq, y_all_seq, seq_info_all = create_breakout_sequences(
        event_manifest_df, all_features_df, ENHANCED_FEATURES, TIME_STEPS
    )
    if len(X_all_seq) == 0: logging.error("No sequences for K-Fold."); return 1
    logging.info(f"Created {len(X_all_seq)} sequences for K-Fold.")

    n_samples, n_steps, n_features = X_all_seq.shape
    X_all_flat = X_all_seq.reshape(n_samples, n_steps * n_features)
    y_all = y_all_seq
    np.nan_to_num(X_all_flat, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    y_all = np.nan_to_num(y_all, nan=1)

    all_portfolio_dfs, all_trades, all_predictions = [], [], []
    t1_dates = pd.to_datetime([info['date'] for info in seq_info_all])
    t1_series = pd.Series(t1_dates + pd.Timedelta(days=BREAKOUT_PERIOD_DAYS), index=range(len(t1_dates)))
    pkf = PurgedKFold(n_splits=N_SPLITS, t1=t1_series, pctEmbargo=0.01)
    num_sequences = len(seq_info_all)

    # Generate feature names for flattened input (needed for importance and saving)
    # Ensure this order matches the reshape operation: (n_samples, time_steps * n_features)
    flat_feature_names = [f"{feat}_t{i}" for i in range(n_steps -1, -1, -1) for feat in ENHANCED_FEATURES] # More intuitive T-N naming
    if len(flat_feature_names) != X_all_flat.shape[1]:
        logging.error(f"FATAL: Flattened feature name count ({len(flat_feature_names)}) doesn't match data shape ({X_all_flat.shape[1]}). Check sequence generation/reshape.")
        return 1

    # --- K-Fold Loop ---
    for fold, (train_idx, test_idx) in enumerate(pkf.split(np.arange(num_sequences))):
        fold_id = fold + 1
        logging.info(f"--- STARTING FOLD {fold_id}/{N_SPLITS} ---")
        gc.collect()
        X_tr_flat, y_tr = X_all_flat[train_idx], y_all[train_idx]
        X_ts_flat, y_ts = X_all_flat[test_idx], y_all[test_idx]
        seq_info_ts = [seq_info_all[i] for i in test_idx]
        if len(X_tr_flat) == 0 or len(X_ts_flat) == 0: logging.warning(f"Skipping fold {fold_id}."); continue

        scaler = MinMaxScaler().fit(X_tr_flat)
        X_tr_s = scaler.transform(X_tr_flat); X_ts_s = scaler.transform(X_ts_flat)
        weights_tr = compute_breakout_sample_weights(y_tr)

        lgbm_params = best_hps.copy()
        lgbm_params.update({'objective': 'multiclass', 'num_class': 3, 'metric': 'auc_mu', 'seed': RANDOM_STATE + fold_id, 'n_jobs': -1, 'verbose': -1}) # Add fold_id to seed
        model = lgb.LGBMClassifier(**lgbm_params)

        logging.info(f"Fold {fold_id}: Fitting LightGBM on {len(X_tr_s)} samples...")
        model.fit(X_tr_s, y_tr, sample_weight=weights_tr, feature_name=flat_feature_names) # Pass feature names

        # Feature Importance Log
        if fold_id == 1: # Log only once
             try:
                 feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': model.feature_name_})
                 feature_imp = feature_imp.sort_values(by="Value", ascending=False)
                 logging.info(f"Top 15 Feature Importances (Fold 1):\n{feature_imp.head(15).to_string()}")
             except Exception as fe_e: logging.warning(f"Could not log feature importance: {fe_e}")

        del X_tr_flat, y_tr, X_tr_s, weights_tr; gc.collect()

        logging.info(f"Fold {fold_id}: Predicting on {len(X_ts_s)} samples...")
        pred_probabilities = model.predict_proba(X_ts_s)
        preds_df = pd.DataFrame(seq_info_ts); preds_df['pred_loss'] = pred_probabilities[:,0]
        preds_df['pred_timeout'] = pred_probabilities[:,1]; preds_df['pred_win'] = pred_probabilities[:,2]
        preds_df['actual_label'] = y_ts - 1; preds_df['fold'] = fold_id
        all_predictions.append(preds_df)

        # Backtest
        logging.info(f"Fold {fold_id}: Running backtest...")
        # ... [ Backtest data preparation logic remains the same ] ...
        test_dates = sorted(list(set(pd.to_datetime(preds_df['date'].unique()))))
        if not test_dates: logging.warning(f"Fold {fold_id}: No prediction dates."); continue
        start_date, end_events = min(test_dates), max(test_dates)
        end_data = end_events + pd.Timedelta(days=strategy_params['holding_period_days'] + 5)
        logging.info(f"Backtest event period fold {fold_id}: {start_date.date()} to {end_events.date()}")
        tickers_fold = preds_df['ticker'].unique()
        bt_cols = ['open', 'high', 'low', 'close', 'raw_atr', 'predicted_regime']
        avail_cols = [c for c in bt_cols if c in all_features_df.columns]
        missing_cols = set(bt_cols) - set(avail_cols);
        if missing_cols: logging.warning(f"Fold {fold_id}: Missing backtest columns: {missing_cols}")
        bt_data = all_features_df.loc[(tickers_fold, slice(start_date, end_data)), avail_cols].copy()
        if bt_data.empty: logging.warning(f"Fold {fold_id}: No raw data."); continue
        bt_data = bt_data.reset_index()
        rename_map = {'date':'Date', 'ticker': 'ticker', **{c: c for c in avail_cols if c not in ['date','ticker']}}
        bt_data.rename(columns=rename_map, inplace=True); bt_data['Date'] = pd.to_datetime(bt_data['Date'])
        bt_data = bt_data.set_index('Date').sort_index()
        cash_fold = all_portfolio_dfs[-1]['total_value'].iloc[-1] if all_portfolio_dfs else INITIAL_PORTFOLIO_CASH
        logging.info(f"Fold {fold_id}: Starting cash: {cash_fold:.2f}")
        preds_df['date'] = pd.to_datetime(preds_df['date'])
        try:
            pf_df, trades_df = run_backtest_on_fold(preds_df, bt_data, strategy_params, cash_fold)
        except Exception as e: logging.error(f"Backtest error fold {fold_id}: {e}", exc_info=True); pf_df, trades_df = pd.DataFrame(), pd.DataFrame()

        if pf_df is not None and not pf_df.empty: all_portfolio_dfs.append(pf_df)
        if trades_df is not None and not trades_df.empty: trades_df['fold'] = fold_id; all_trades.append(trades_df)
        logging.info(f"--- COMPLETED FOLD {fold_id}/{N_SPLITS} ---")
        del X_ts_flat, y_ts, X_ts_s, pred_probabilities, preds_df, model, scaler, pf_df, trades_df, train_idx, test_idx, bt_data; gc.collect()

    # --- End K-Fold ---
    del X_all_seq, y_all_seq, X_all_flat, y_all; gc.collect()

    # --- Phase 6: Final Evaluation ---
    # (Remains unchanged)
    logging.info("Phase 6: Final Evaluation")
    # ... [ Code from previous main.py for Phase 6 ] ...
    final_preds_df, final_trades_df, final_pf_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    fin_metrics, stat_metrics = {}, {}
    # ... (Concatenate predictions, trades, portfolios; calculate metrics; save results/plots) ...
    if all_predictions:
        final_preds_df = pd.concat(all_predictions).sort_values(by=['date', 'ticker']).reset_index(drop=True)
        pred_fn = PREDICTIONS_FILENAME_TPL.format(run_id=run_id)
        try: final_preds_df.to_csv(pred_fn, index=False); logging.info(f"Saved predictions ({len(final_preds_df)}) to {pred_fn}")
        except Exception as e: logging.error(f"Failed to save predictions: {e}")
        try:
             y_true = final_preds_df['actual_label'] + 1; y_proba = final_preds_df[['pred_loss','pred_timeout','pred_win']].values
             stat_metrics = inspect_label_quality(y_true.values, y_proba, save_prefix=f'backtests/final_oos_{run_id}')
        except Exception as e: logging.error(f"Failed to calculate model stats: {e}")
    else: logging.warning("No predictions generated.")

    if all_trades:
        final_trades_df = pd.concat(all_trades).sort_values(by=['exit_date', 'ticker']).reset_index(drop=True)
        if 'exit_date' in final_trades_df.columns: final_trades_df['exit_date'] = pd.to_datetime(final_trades_df['exit_date'])
        if 'entry_date' in final_trades_df.columns: final_trades_df['entry_date'] = pd.to_datetime(final_trades_df['entry_date'])
        trades_fn = TRADES_FILENAME_TPL.format(run_id=run_id)
        try: final_trades_df.to_csv(trades_fn, index=False); logging.info(f"Saved trades ({len(final_trades_df)}) to {trades_fn}")
        except Exception as e: logging.error(f"Failed to save trades: {e}")
    else: logging.warning("No trades generated.")

    if all_portfolio_dfs:
        try:
            final_pf_df = pd.concat(all_portfolio_dfs).sort_index(); final_pf_df = final_pf_df[~final_pf_df.index.duplicated(keep='last')]
            fin_metrics = calculate_financial_metrics(final_pf_df['total_value'], final_trades_df, INITIAL_PORTFOLIO_CASH)
            if not final_trades_df.empty and regime_map:
                 try: regime_perf = analyze_performance_by_regime(final_trades_df, regime_map); fin_metrics['performance_by_regime'] = regime_perf
                 except Exception as e: logging.error(f"Regime analysis failed: {e}")
            logging.info("--- Final Financial Performance ---")
            for k, v in fin_metrics.items():
                 if k != 'performance_by_regime': logging.info(f"  {k}: {v}")
            plot_equity_curve(final_pf_df['total_value'], filename=EQUITY_CURVE_FILENAME_TPL.format(run_id=run_id), run_id=run_id)
        except Exception as e: logging.error(f"Error analyzing final results: {e}", exc_info=True)
    else: logging.warning("No portfolio data. Skipping financial analysis.")

    save_results_summary(
        json_filename=JSON_SUMMARY_FILENAME_TPL.format(run_id=run_id),
        run_id=run_id, strategy_params=strategy_params,
        financial_metrics=fin_metrics, statistical_metrics=stat_metrics,
        best_hps=best_hps
    )


    # --- Phase 7: Retrain Final LightGBM Model ---
    if args.save_model:
        logging.info("Phase 7: Retraining final LightGBM model...")
        if args.offline: logging.warning("Offline mode.")
        logging.info("Creating sequences for final model...")
        # Regenerate sequences or reuse X_all_flat, y_all if available
        # For safety and less memory pressure, let's regenerate
        X_final_seq, y_final, _ = create_breakout_sequences(
            event_manifest_df, all_features_df, ENHANCED_FEATURES, TIME_STEPS
        )

        if len(X_final_seq) == 0: logging.error("No sequences available. Skipping save.")
        else:
            final_scaler = None; final_model = None # Initialize
            X_final_flat = None; X_final_s = None; weights_final = None # Initialize
            try:
                gc.collect()
                logging.info("Scaling final data (flattened)...")
                n_samples, n_steps, n_features = X_final_seq.shape
                X_final_flat = X_final_seq.reshape(n_samples, n_steps * n_features)
                y_final = np.nan_to_num(y_final, nan=1)
                np.nan_to_num(X_final_flat, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

                final_scaler = MinMaxScaler().fit(X_final_flat)
                X_final_s = final_scaler.transform(X_final_flat)
                weights_final = compute_breakout_sample_weights(y_final)

                # Build final model
                final_lgbm_params = best_hps.copy()
                final_lgbm_params.update({'objective': 'multiclass', 'num_class': 3, 'metric': 'auc_mu',
                                          'seed': RANDOM_STATE, 'n_jobs': -1, 'verbose': -1})
                final_model = lgb.LGBMClassifier(**final_lgbm_params)

                logging.info(f"Fitting final LightGBM model on {len(X_final_s)}...")
                # Pass feature names for potential saving within model file if supported
                final_model.fit(X_final_s, y_final, sample_weight=weights_final, feature_name=flat_feature_names)

                logging.info("Saving final LightGBM model and artifacts.")
                fn_model = BREAKOUT_MODEL_FILENAME; fn_scaler = BREAKOUT_SCALER_FILENAME; fn_meta = BREAKOUT_METADATA_FILENAME

                # Save LightGBM model (native method recommended)
                final_model.booster_.save_model(fn_model)
                # Or use joblib: joblib.dump(final_model, fn_model)

                with open(fn_scaler, 'wb') as f: pickle.dump(final_scaler, f)
                # Save metadata including feature names
                metadata = {'features': flat_feature_names, # Save flattened names used by model
                            'original_features': ENHANCED_FEATURES, # Keep original list
                            'time_steps': TIME_STEPS,
                            'best_hps': best_hps,
                            'strategy_params': strategy_params}
                with open(fn_meta, 'wb') as f: pickle.dump(metadata, f)
                logging.info(f"Final artifacts saved:\n  Model: {fn_model}\n  Scaler: {fn_scaler}\n  Meta: {fn_meta}")

            except Exception as e: logging.error(f"Final model save failed: {e}", exc_info=True)
            # No finally block needed here as variables are local to the 'else'

    else: logging.info("Skipping final model save.")

    end_time = time.time()
    logging.info(f"--- RUN {run_id} COMPLETE --- (Total Time: {(end_time - start_time)/60:.2f} minutes)")
    return 0 # Success

# --- Argument Parser ---
def setup_arg_parser():
    parser = argparse.ArgumentParser(description="V6 Gated Walk-Forward Breakout Model Trainer (LightGBM).") # Updated desc
    parser.add_argument('--run-id', type=str, default=f"v6_lgbm_{datetime.now().strftime('%Y%m%d_%H%M%S')}", help="Unique ID.") # Updated default
    parser.add_argument('--offline', action='store_true', help="Offline mode.")
    parser.add_argument('--force-retune', action='store_true', help="Force Optuna re-run.") # Updated help text
    parser.add_argument('--force-rebuild-features', action='store_true', help="Force feature rebuild.")
    parser.add_argument('--save-model', action='store_true', help="Save final model.")
    # Strategy Params
    parser.add_argument('--confidence', type=float, default=None, help=f"Win prob threshold (default: {DEFAULT_STRATEGY_PARAMS['confidence_threshold']}).")
    parser.add_argument('--risk-per-trade', type=float, default=None, help=f"Risk % (default: {DEFAULT_STRATEGY_PARAMS['risk_per_trade_percent']}).")
    parser.add_argument('--hold-period', type=int, default=None, help=f"Max hold days (default: {DEFAULT_STRATEGY_PARAMS['holding_period_days']}).")
    parser.add_argument('--regime-filter', action=argparse.BooleanOptionalAction, default=DEFAULT_STRATEGY_PARAMS['use_regime_filter'], help="Regime filter.")
    parser.add_argument('--use-trailing-stop', action=argparse.BooleanOptionalAction, default=DEFAULT_STRATEGY_PARAMS['use_trailing_stop'], help="Trailing stop.")
    parser.add_argument('--atr-stop-mult', type=float, default=None, help=f"ATR multiplier (default: {DEFAULT_STRATEGY_PARAMS['atr_stop_multiplier']}).")
    parser.add_argument('--stop-loss', type=float, default=None, help=f"Fixed stop % (default: {DEFAULT_STRATEGY_PARAMS['stop_loss_percent']}).")
    return parser

# --- Main Guard ---
if __name__ == "__main__":
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()
    exit_code = 1
    try:
        exit_code = main(args)
    except Exception as e:
         logging.error(f"Unhandled exception: {e}", exc_info=True)
    finally:
         logging.info(f"Script finished with exit code {exit_code}")
         logging.shutdown()
         exit(exit_code)