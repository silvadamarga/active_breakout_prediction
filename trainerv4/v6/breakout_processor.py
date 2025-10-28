#!/usr/bin/env python3
"""
v6 - Breakout Model Data Processor
Handles feature engineering (Blueprint + Enhanced) and sequence creation.
Calculates RS vs Bonds, Breadth, SMA Ratio, ATR Volatility, ROC.
"""

import logging
import numpy as np
import pandas as pd
import pandas_ta as ta
import tensorflow as tf # Keep for tf.data if using Keras
from typing import List, Dict, Tuple, Any
from sklearn.preprocessing import MinMaxScaler

# Import config and utils
from .config import (
    MIN_HISTORY_REQUIREMENT, SECTOR_ETF_MAP, BLUEPRINT_FEATURES, # Keep blueprint
    ENHANCED_FEATURES, # Use enhanced list
    BREAKOUT_PERIOD_DAYS, TIME_STEPS
)
# --- Import DataManager locally if needed ---


# --- Helper Function ---
def get_slope(array):
    """Calculates the slope of a 1D array using linear regression."""
    y = np.nan_to_num(array) # Ensure no NaNs before calculation
    valid_points = ~np.isnan(y) # Find where original was not NaN
    if valid_points.sum() < 2: return 0.0 # Need at least 2 non-NaN points
    x = np.arange(len(y))
    try:
        # Fit only on the valid points
        slope = np.polyfit(x[valid_points], y[valid_points], 1)[0]
    except (np.linalg.LinAlgError, ValueError) as e:
        # logging.debug(f"Slope calculation failed: {e}") # Optional debug log
        slope = 0.0 # Handle singular matrix or other fitting errors
    return slope if np.isfinite(slope) else 0.0 # Ensure finite output

def _calculate_manual_bbands(close_series, length=20, std=2.0):
    """Manually calculates Bollinger Bands to bypass pandas-ta issues."""
    if close_series is None or close_series.empty or close_series.isna().all(): return None, None
    try:
        middle_band = ta.sma(close_series, length=length)
        stdev = ta.stdev(close_series, length=length)
        if middle_band is None or stdev is None or middle_band.isna().all() or stdev.isna().all(): return None, None
        upper_band = middle_band + (stdev * std)
        lower_band = middle_band - (stdev * std)
        return upper_band, lower_band
    except Exception as e:
        # logging.debug(f"Error calculating BBands: {e}") # Optional debug log
        return None, None


def process_data_for_ticker(ticker: str, df: pd.DataFrame, context_dfs: dict, ticker_to_sector: dict) -> pd.DataFrame | None:
    """
    Generates enhanced features, screener conditions, and triple-barrier labels
    for a SINGLE stock. Returns only necessary columns or None on failure.
    """
    if df is None or len(df) < MIN_HISTORY_REQUIREMENT:
        logging.debug(f"Skipping {ticker}: Not enough initial data ({len(df) if df is not None else 0} rows < {MIN_HISTORY_REQUIREMENT})")
        return None

    df = df.copy().sort_index()

    # --- 1. Rename and Validate Columns ---
    df.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low',
        'Close': 'close', 'Volume': 'volume'
    }, inplace=True, errors='ignore')

    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        logging.warning(f"Skipping {ticker}: Missing one of required OHLCV columns.")
        return None

    # Convert essential columns to numeric, coercing errors
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Drop rows if ANY essential column is NaN after conversion
    df.dropna(subset=required_cols, inplace=True)

    if len(df) < MIN_HISTORY_REQUIREMENT:
        logging.debug(f"Skipping {ticker}: Not enough valid OHLCV data after NaN cleanup ({len(df)} rows < {MIN_HISTORY_REQUIREMENT}).")
        return None

    # --- 2. Merge Context Data ---
    # Merge context data safely using .get() for dict keys
    def safe_merge(df_left, context_key, cols_to_merge):
        context_df = context_dfs.get(context_key)
        if context_df is not None:
            existing_cols = [c for c in cols_to_merge if c in context_df.columns]
            if existing_cols:
                return df_left.merge(context_df[existing_cols], left_index=True, right_index=True, how='left')
            else: logging.debug(f"Context key '{context_key}' found but columns '{cols_to_merge}' missing.")
        # else: logging.debug(f"Context key '{context_key}' not found.")
        return df_left # Return original df if merge fails

    df = safe_merge(df, 'SPY', ['SPY_Close'])
    sector = ticker_to_sector.get(ticker)
    if sector and sector in SECTOR_ETF_MAP:
        etf = SECTOR_ETF_MAP[sector]
        df = safe_merge(df, etf, ['Sector_Close'])
    df = safe_merge(df, 'VIX', ['VIX_Close', 'VIX_SMA_5', 'VIX_SMA_20'])
    df = safe_merge(df, 'TLT', ['TLT_Close'])
    df = safe_merge(df, 'breadth', ['pct_above_50d', 'pct_above_200d'])
    df = safe_merge(df, 'predicted_regimes', ['predicted_regime'])

    # Fill gaps AFTER merging all context data (consider limit for large gaps)
    # df.ffill(limit=5, inplace=True) # Limit fill to avoid carrying stale data too far?
    df.ffill(inplace=True)
    close = df['close'] # Use after potential ffill

    # --- 3. Calculate Blueprint & Enhanced Features ---
    # Use try-except block for robustness during TA calculations
    try:
        # Volatility
        bbu, bbl = _calculate_manual_bbands(close, length=20, std=2.0)
        if bbu is not None and bbl is not None:
            bbw = (bbu - bbl).replace([np.inf, -np.inf], np.nan) # Handle potential inf
            df['BBW_norm'] = (bbw / close).replace([np.inf, -np.inf], np.nan)
            df['BBW_6mo_rank'] = df['BBW_norm'].rolling(126, min_periods=30).rank(pct=True)
        else: df['BBW_norm'], df['BBW_6mo_rank'] = np.nan, np.nan

        df['raw_atr'] = ta.atr(df['high'], df['low'], close, length=14)
        df['ATR_norm'] = (df['raw_atr'] / close).replace([np.inf, -np.inf], np.nan)
        df['ATR_6mo_rank'] = df['ATR_norm'].rolling(126, min_periods=30).rank(pct=True)
        df['ATR_norm_std_20d'] = df['ATR_norm'].rolling(20, min_periods=10).std() # Added
        df['Choppiness_14d'] = ta.chop(df['high'], df['low'], close, length=14)

        # Momentum/Trend
        adx_df = ta.adx(df['high'], df['low'], close, length=14)
        df['ADX_14d'] = adx_df['ADX_14'] if adx_df is not None and 'ADX_14' in adx_df.columns else np.nan

        sma_20 = ta.sma(close, 20)
        sma_50 = ta.sma(close, 50)
        if sma_50 is not None: df['SMA_50_slope'] = sma_50.rolling(10, min_periods=5).apply(get_slope, raw=True).fillna(0)
        else: df['SMA_50_slope'] = 0.0
        # Calculate SMA Ratio safely
        if sma_20 is not None and sma_50 is not None:
            df['SMA_20_50_Ratio'] = (sma_20 / sma_50.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan) # Added
        else: df['SMA_20_50_Ratio'] = np.nan

        df['ROC_20d'] = ta.roc(close, length=20) # Added

        # Volume
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).clip(lower=0)
        obv = ta.obv(close, df['volume'])
        if obv is not None: df['OBV_slope_20d'] = obv.rolling(20, min_periods=10).apply(get_slope, raw=True).fillna(0)
        else: df['OBV_slope_20d'] = 0.0
        vol_sma_50 = ta.sma(df['volume'], 50)
        df['Volume_vs_Avg'] = (df['volume'] / vol_sma_50.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    except Exception as e:
        logging.error(f"Error calculating TA features for {ticker}: {e}", exc_info=False)
        # Allow continuation for non-TA features, NaNs will be handled later

    # Relative Strength (calculate safely, using .get() with default)
    def calculate_rs_slope(series1, series2_name, lookback=50, min_p=20):
        series2 = df.get(series2_name) # Safely get the series
        if series1 is None or series2 is None or series1.empty or series2.empty: return pd.Series(0.0, index=df.index)
        # Ensure alignment before division
        s1_aligned, s2_aligned = series1.align(series2, join='left')
        rs = (s1_aligned / s2_aligned.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        return rs.rolling(lookback, min_periods=min_p).apply(get_slope, raw=True).fillna(0)

    df['RS_vs_SPX_slope'] = calculate_rs_slope(close, 'SPY_Close')
    df['RS_vs_Sector_slope'] = calculate_rs_slope(close, 'Sector_Close')
    df['RS_vs_Bonds_slope'] = calculate_rs_slope(close, 'TLT_Close') # Added

    # Market Context & Breadth Features (these come from context_dfs)
    df['VIX_level'] = df.get('VIX_Close')
    vix_sma_20 = df.get('VIX_SMA_20')
    df['VIX_MA_ratio'] = (df.get('VIX_SMA_5') / vix_sma_20.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan) if vix_sma_20 is not None else np.nan
    # 'pct_above_50d', 'pct_above_200d' are already in df if merged successfully

    # Ensure predicted_regime defaults if missing
    df['predicted_regime'] = df.get('predicted_regime', 2.0).fillna(2.0)

    # --- 4. Generate Labels (Triple-Barrier) ---
    upper_mult = 3.0; lower_mult = 1.5; time_limit_days = BREAKOUT_PERIOD_DAYS
    labels = pd.Series(index=df.index, data=np.nan, dtype=float)
    prices_arr = df['close'].values; high_arr = df['high'].values
    low_arr = df['low'].values; atr_arr = df.get('raw_atr', pd.Series(dtype=float)).values # Use .get()
    indices = df.index

    for i in range(len(prices_arr) - time_limit_days):
        current_atr = atr_arr[i]; entry_price = prices_arr[i]
        if pd.isna(current_atr) or current_atr <= 1e-6 or pd.isna(entry_price): continue
        upper_barrier = entry_price + (current_atr * upper_mult)
        lower_barrier = entry_price - (current_atr * lower_mult)
        hit_upper, hit_lower = False, False
        for j in range(1, time_limit_days + 1):
            if i + j >= len(high_arr): break # Check array bounds
            future_high = high_arr[i + j]; future_low = low_arr[i + j]
            if pd.isna(future_high) or pd.isna(future_low): break
            if future_high >= upper_barrier: labels.loc[indices[i]] = 1.0; hit_upper = True; break
            if future_low <= lower_barrier: labels.loc[indices[i]] = -1.0; hit_lower = True; break
        if not hit_upper and not hit_lower: labels.loc[indices[i]] = 0.0
    df['label'] = labels

    # --- 5. Generate Screener Conditions ---
    # Calculate screener components safely using .get()
    spy_close = df.get('SPY_Close')
    spy_sma_50 = ta.sma(spy_close, 50) if spy_close is not None else None
    spy_sma_200 = ta.sma(spy_close, 200) if spy_close is not None else None
    df['macro_trend_pass'] = (spy_close > spy_sma_50) & (spy_close > spy_sma_200) if spy_close is not None and spy_sma_50 is not None and spy_sma_200 is not None else False

    vix_level = df.get('VIX_level', 100).fillna(100)
    vix_ma_ratio = df.get('VIX_MA_ratio', 2.0).fillna(2.0)
    df['macro_vol_pass'] = (vix_level < 25) & (vix_ma_ratio < 1.0)

    # Use rs_spx calculated earlier
    rs_spx = (close / df.get('SPY_Close', np.nan)).replace([np.inf, -np.inf], np.nan)
    rs_high_252d = rs_spx.rolling(252, min_periods=100).max()
    df['stock_rs_near_high'] = (rs_spx >= rs_high_252d * 0.98) & rs_spx.notna()

    bbw_norm = df.get('BBW_norm')
    if bbw_norm is not None:
        bbw_6mo_low = bbw_norm.rolling(126, min_periods=30).min()
        df['bbw_squeeze'] = (bbw_norm <= bbw_6mo_low * 1.05) & bbw_norm.notna()
    else: df['bbw_squeeze'] = False

    atr_norm = df.get('ATR_norm')
    if atr_norm is not None:
        atr_6mo_low = atr_norm.rolling(126, min_periods=30).min()
        df['atr_low'] = (atr_norm <= atr_6mo_low * 1.05) & atr_norm.notna()
    else: df['atr_low'] = False

    df['is_choppy'] = df.get('Choppiness_14d', 0).fillna(0) > 61.8

    # OBV rising check (simplified: positive slope is sufficient)
    df['obv_rising'] = df.get('OBV_slope_20d', 0).fillna(0) > 0

    # --- 6. Combine Screener Conditions ---
    conditions = [
        df.get('macro_trend_pass', False), df.get('macro_vol_pass', False),
        df.get('stock_rs_near_high', False),
        (df.get('bbw_squeeze', False) | df.get('atr_low', False)), # Squeeze OR low ATR
        df.get('obv_rising', False)
    ]
    # Ensure all conditions are Series aligned with df's index
    aligned_conditions = []
    for cond in conditions:
        if isinstance(cond, pd.Series): aligned_conditions.append(cond.reindex(df.index).fillna(False))
        elif isinstance(cond, bool): aligned_conditions.append(pd.Series(cond, index=df.index))
        else: aligned_conditions.append(pd.Series(False, index=df.index)) # Default to False if invalid type

    df['PASS_SCREENER'] = pd.concat(aligned_conditions, axis=1).all(axis=1)


    # --- 7. Final Cleanup and Column Selection ---
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Define columns needed: features + label + screener + backtest info
    # Use ENHANCED_FEATURES from config
    final_cols_needed = ENHANCED_FEATURES + ['label', 'PASS_SCREENER', 'raw_atr', 'open', 'high', 'low', 'close', 'predicted_regime']
    final_cols_exist = sorted(list(set([col for col in final_cols_needed if col in df.columns]))) # Keep unique and sort
    df = df[final_cols_exist]

    # Drop rows where essential features for the *model* or the screener flag are NaN
    essential_model_cols = ENHANCED_FEATURES + ['PASS_SCREENER']
    missing_essential = [col for col in essential_model_cols if col not in df.columns]
    if missing_essential:
         logging.warning(f"Skipping {ticker}: Missing essential cols post-calculation: {missing_essential}")
         return None

    # How many NaNs before drop?
    nan_before = df[essential_model_cols].isna().any(axis=1).sum()
    df.dropna(subset=essential_model_cols, inplace=True)
    nan_after = df[essential_model_cols].isna().any(axis=1).sum()
    if nan_before > 0: logging.debug(f"{ticker}: Dropped {nan_before} rows with NaNs in essential columns.")
    if nan_after > 0: logging.warning(f"{ticker}: {nan_after} NaNs remain in essential columns AFTER dropna? Check logic.")


    if df.empty:
         logging.debug(f"Skipping {ticker}: No rows remained after final dropna.")
         return None

    # Final check: ensure all ENHANCED_FEATURES are present before returning
    missing_final = [f for f in ENHANCED_FEATURES if f not in df.columns]
    if missing_final:
        logging.warning(f"Skipping {ticker}: Final DataFrame missing features: {missing_final}")
        return None


    df.index.name = 'date'
    return df


# --- Orchestrator (build_event_manifest) ---
# (Remains largely the same as previous version, ensures process_data_for_ticker is called)
def build_event_manifest(
    data_manager: Any, base_tickers: List[str],
    context_dfs: Dict[str, pd.DataFrame], ticker_to_sector: Dict[str, str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Orchestrates feature creation and event manifest generation."""
    if 'DataManager' not in globals(): from .data_manager import DataManager

    all_feature_dfs, event_manifest_data = [], []
    logging.info(f"Processing features for {len(base_tickers)} tickers...")
    all_stock_data = data_manager.fetch_all_data(base_tickers)
    processed_count = 0

    for i, ticker in enumerate(base_tickers):
        if i % 50 == 0 and i > 0: logging.info(f"  ...processing ticker {i}/{len(base_tickers)} ({ticker}) - {processed_count} valid")
        if ticker not in all_stock_data: continue

        processed_df = process_data_for_ticker(ticker, all_stock_data[ticker], context_dfs, ticker_to_sector)
        if processed_df is None or processed_df.empty: continue
        processed_count += 1

        event_df = processed_df[(processed_df['PASS_SCREENER'] == True) & (processed_df['label'].notna())].copy()
        if not event_df.empty:
            for event_date, row in event_df.iterrows():
                event_manifest_data.append({'date': event_date, 'ticker': ticker, 'label': row['label']})

        all_feature_dfs.append(processed_df.assign(ticker=ticker))

    logging.info(f"Finished processing. Found valid data for {processed_count}/{len(base_tickers)} tickers.")
    if not all_feature_dfs: logging.error("No valid processed data."); return pd.DataFrame(), pd.DataFrame()

    all_features_df = pd.concat(all_feature_dfs)
    event_manifest_df = pd.DataFrame(event_manifest_data)

    # Set multi-index
    if not all_features_df.empty:
        df_reset = all_features_df.reset_index() # Get 'date' as column
        if 'ticker' in df_reset.columns and 'date' in df_reset.columns:
            df_reset['date'] = pd.to_datetime(df_reset['date'])
            all_features_df = df_reset.set_index(['ticker', 'date']).sort_index()
        else: logging.error("Cannot set MultiIndex on features: missing columns."); return pd.DataFrame(), pd.DataFrame()

    if not event_manifest_df.empty and 'date' in event_manifest_df.columns and 'ticker' in event_manifest_df.columns:
        event_manifest_df['date'] = pd.to_datetime(event_manifest_df['date'])
        event_manifest_df = event_manifest_df.set_index(['ticker', 'date']).sort_index()

    return all_features_df, event_manifest_df

# --- Sequence Generation (create_breakout_sequences) ---
# (Remains largely the same, ensure it uses the 'features' argument correctly)
def create_breakout_sequences(
    event_manifest: pd.DataFrame, all_features_df: pd.DataFrame,
    features: List[str], time_steps: int
):
    """Builds sequences using the specified feature list."""
    X, y, seq_info = [], [], []
    if not isinstance(all_features_df.index, pd.MultiIndex): logging.error("Features DF needs MultiIndex."); return np.array([]), np.array([]), []
    if event_manifest.empty: logging.warning("Empty manifest."); return np.array([]), np.array([]), []
    if not isinstance(event_manifest.index, pd.MultiIndex):
        if 'ticker' in event_manifest.columns and 'date' in event_manifest.columns:
             event_manifest = event_manifest.set_index(['ticker', 'date']).sort_index()
        else: logging.error("Manifest lacks index/cols."); return np.array([]), np.array([]), []

    logging.info(f"Generating sequences for {len(event_manifest)} events using {len(features)} features...")
    skipped = {'history': 0, 'nan': 0, 'lookup': 0, 'feature_mismatch': 0}

    grouped_features = all_features_df.groupby(level='ticker')
    available_tickers = set(grouped_features.groups.keys())

    # Check if requested features are even available in the main dataframe
    base_features_available = [f for f in features if f in all_features_df.columns]
    if len(base_features_available) != len(features):
         logging.error(f"FATAL: Requested features {set(features) - set(base_features_available)} not found in all_features_df columns.")
         return np.array([]), np.array([]), []


    for idx, row in event_manifest.iterrows():
        ticker, event_date = idx; event_label = row['label']
        if ticker not in available_tickers: skipped['lookup'] += 1; continue

        try:
            ticker_features_mi = grouped_features.get_group(ticker)
            ticker_features = ticker_features_mi.reset_index(level='ticker', drop=True).sort_index()
            event_iloc = ticker_features.index.get_loc(event_date)

            if event_iloc < time_steps: skipped['history'] += 1; continue

            # Select only the requested features for the sequence slice
            sequence_data = ticker_features.iloc[event_iloc - time_steps : event_iloc][features].values

            # Shape check (should match if base check passed, but good safeguard)
            if sequence_data.shape[1] != len(features):
                skipped['feature_mismatch'] += 1
                continue

            # Check for NaNs/Infs
            if not np.isfinite(sequence_data).all():
                skipped['nan'] += 1; continue

            X.append(sequence_data.astype(np.float32))
            y.append(int(event_label) + 1) # Map -1,0,1 -> 0,1,2
            seq_info.append({'ticker': ticker, 'date': event_date})

        except KeyError: skipped['lookup'] += 1 # Date not found after reset? Unlikely but possible.
        except Exception as e: logging.error(f"Seq gen error {ticker} @ {event_date}: {e}", exc_info=False)

    # Report skipped counts
    for reason, count in skipped.items():
        if count > 0: logging.warning(f"Skipped {count} sequences due to: {reason}")

    if not X: logging.error("No sequences generated."); return np.array([]), np.array([]), []

    logging.info(f"Successfully generated {len(X)} sequences.")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), seq_info

# --- TF Dataset Creator (create_tf_dataset) ---
# (Unchanged)
def create_tf_dataset(X, y_categorical, sample_weights, batch_size, shuffle=False):
    """Wraps numpy arrays in a memory-efficient tf.data.Dataset (for Keras)."""
    dataset = tf.data.Dataset.from_tensor_slices((X, y_categorical, sample_weights))
    if shuffle:
        buffer_size = min(len(X), 10000)
        dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset