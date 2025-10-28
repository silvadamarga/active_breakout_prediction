#!/usr/bin/env python3
"""
Generates a table of current model confidence for all S&P 500 stocks.

This script loads the trained regime model (Keras) and the final breakout model
(LightGBM) from the v6 project. It re-runs the entire data
synchronization and feature engineering pipeline to get the most up-to-date
feature vectors for all S&P 500 stocks, then predicts their confidence.
"""

import os
import logging
import pickle
import json
import warnings
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
import tensorflow as tf

# Suppress warnings from libraries
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('yfinance').setLevel(logging.ERROR)
logging.getLogger('pandas_ta').setLevel(logging.ERROR)

# Add the v6 directory to the path if it's not already
# This assumes the script is run from the parent directory of v6
script_dir = os.path.dirname(os.path.abspath(__file__))
# We add the parent directory to the path so that 'from v6...' works
if script_dir not in sys.path:
    sys.path.insert(0, script_dir) 

try:
    # --- Project Module Imports ---
    # Updated all imports from 'v6_final' to 'v6'
    from v6.config import (
        DB_CACHE_FILE, TIME_STEPS, ENHANCED_FEATURES, SECTOR_ETF_MAP,
        REGIME_TICKERS, BREAKOUT_MODEL_FILENAME, BREAKOUT_SCALER_FILENAME,
        BREAKOUT_METADATA_FILENAME, LOG_FILENAME_TPL
    )
    from v6.data_manager import (
        DataManager, get_sp500_tickers_cached, get_or_create_sector_map
    )
    from v6.regime_model import (
        load_regime_model_and_artifacts, build_and_predict_regime_df
    )
    from v6.breakout_processor import build_event_manifest
    from v6.main import setup_logging # Correctly imported from main
    
    # We need pandas-ta for the feature engineering step
    import pandas_ta as ta
except ImportError as e:
    print(f"Error: Could not import project modules: {e}")
    print("Please ensure this script is in the parent directory of your 'v6' folder.")
    print("Also ensure all required libraries (pandas, tensorflow, lightgbm, etc.) are installed.")
    sys.exit(1)


def generate_confidence_report():
    """
    Main function to run the entire pipeline and generate the confidence report.
    """
    run_id = f"confidence_report_{datetime.now().strftime('%Y%m%d')}"
    
    # Use the setup_logging function from the project
    setup_logging(run_id) 
    
    logging.info(f"--- Starting Confidence Report: {run_id} ---")

    # --- 1. Load All Models and Scalers ---
    logging.info("Loading all models and artifacts...")
    try:
        # Load Regime Model (Keras)
        regime_model, regime_scaler, regime_features, regime_map = load_regime_model_and_artifacts()
        if regime_model is None:
            logging.error("Failed to load regime model. Aborting.")
            return

        # Load Breakout Model (LightGBM)
        with open(BREAKOUT_SCALER_FILENAME, 'rb') as f:
            breakout_scaler = pickle.load(f)
        
        with open(BREAKOUT_METADATA_FILENAME, 'rb') as f:
            breakout_meta = pickle.load(f)
        
        breakout_model = lgb.Booster(model_file=BREAKOUT_MODEL_FILENAME)
        flat_feature_names = breakout_meta.get('features') # Get feature names model was trained on
        if not flat_feature_names:
             logging.error("Failed to load feature names from breakout metadata. Aborting.")
             return

        logging.info("All models and artifacts loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading models: {e}. Ensure all artifact files exist.", exc_info=True)
        return

    # --- 2. Get S&P 500 Tickers ---
    logging.info("Fetching S&P 500 tickers...")
    # Force online fetch to get the current list
    sp500_tickers = get_sp500_tickers_cached(offline_mode=False) 
    if not sp500_tickers:
        logging.error("Could not fetch S&P 500 tickers. Aborting.")
        return
    logging.info(f"Found {len(sp500_tickers)} S&P 500 tickers.")

    # --- 3. Initialize DataManager ---
    # We must run in online mode to get the latest data
    data_manager = DataManager(DB_CACHE_FILE, offline_mode=False)
    
    # --- 4. Get Sector Map ---
    ticker_to_sector = get_or_create_sector_map(sp500_tickers, offline_mode=False)
    if not ticker_to_sector:
        logging.warning("Could not fetch sector map. Sector-based features will be missing.")

    # --- 5. Run Regime Prediction Pipeline ---
    logging.info("Generating current market regime predictions...")
    # This fetches all regime tickers and predicts the full history
    predicted_regimes_df = build_and_predict_regime_df(
        data_manager, regime_model, regime_scaler, regime_features, offline_mode=False
    )
    if predicted_regimes_df.empty:
        logging.error("Failed to generate regime predictions. Aborting.")
        return
    logging.info("Regime prediction complete.")

    # --- 6. Run Full Feature Engineering Pipeline ---
    logging.info("Building feature database for all S&P 500 stocks...")
    
    # This logic is copied from main.py to build the 'context_dfs'
    sector_etfs = list(set(SECTOR_ETF_MAP.values()))
    context_tickers = list(set(['SPY', '^VIX', 'TLT'] + sector_etfs + REGIME_TICKERS))
    context_data_map = data_manager.fetch_all_data(context_tickers)
    
    # We must fetch breadth data manually here
    context_dfs = {'predicted_regimes': predicted_regimes_df}
    
    try:
        logging.info("Calculating market breadth...")
        sp500_data_map = data_manager.fetch_all_data(sp500_tickers)
        above_50d_list, above_200d_list = [], []
        spy_df_ref = context_data_map.get('SPY')
        if spy_df_ref is not None:
            common_index = spy_df_ref.index.sort_values()
            for ticker in sp500_tickers:
                df = sp500_data_map.get(ticker)
                if df is not None and not df.empty and 'Close' in df.columns:
                    df_aligned = df.reindex(common_index).ffill()
                    if df_aligned['Close'].notna().sum() > 50:
                        sma_50 = ta.sma(df_aligned['Close'], 50)
                        above_50d_list.append((df_aligned['Close'] > sma_50) & df_aligned['Close'].notna() & sma_50.notna())
                    if df_aligned['Close'].notna().sum() > 200:
                        sma_200 = ta.sma(df_aligned['Close'], 200)
                        above_200d_list.append((df_aligned['Close'] > sma_200) & df_aligned['Close'].notna() & sma_200.notna())
            if above_50d_list: 
                context_dfs['breadth'] = pd.DataFrame({
                    'pct_above_50d': pd.concat(above_50d_list, axis=1).astype(float).mean(axis=1) * 100,
                    'pct_above_200d': pd.concat(above_200d_list, axis=1).astype(float).mean(axis=1) * 100
                })
                logging.info("Breadth features calculated.")
        else:
            logging.warning("SPY data not found, cannot calculate breadth.")
    except Exception as e:
        logging.error(f"Error calculating breadth: {e}", exc_info=True)

    # Add other context tickers
    def process_context_ticker(ticker_name, col_name, cols_to_keep):
        if ticker_name in context_data_map:
            df = context_data_map[ticker_name].copy()
            rename_dict = {'Close': col_name}
            df.rename(columns=rename_dict, inplace=True)
            if ticker_name == '^VIX':
                df['VIX_SMA_5'] = ta.sma(df[col_name], 5)
                df['VIX_SMA_20'] = ta.sma(df[col_name], 20)
                cols_to_keep.extend(['VIX_SMA_5', 'VIX_SMA_20'])
            context_dfs[ticker_name.replace('^','')] = df[cols_to_keep]
        
    process_context_ticker('SPY', 'SPY_Close', ['SPY_Close'])
    process_context_ticker('^VIX', 'VIX_Close', ['VIX_Close'])
    process_context_ticker('TLT', 'TLT_Close', ['TLT_Close'])
    for etf in sector_etfs:
        process_context_ticker(etf, 'Sector_Close', ['Sector_Close'])

    # Now, run the main feature build
    all_features_df, _ = build_event_manifest(
        data_manager, sp500_tickers, context_dfs, ticker_to_sector
    )

    if all_features_df.empty:
        logging.error("Failed to build feature database. Aborting.")
        return
    logging.info("Feature database built successfully.")

    # --- 7. Extract Latest Sequences ---
    logging.info("Extracting latest feature sequences for all stocks...")
    results = []
    grouped = all_features_df.groupby(level='ticker')

    for ticker, stock_df in grouped:
        stock_df = stock_df.droplevel(0).sort_index() # Drop ticker index, sort by date
        
        if len(stock_df) >= TIME_STEPS:
            # Get the last TIME_STEPS rows
            latest_sequence_df = stock_df.iloc[-TIME_STEPS:]
            
            # Get the features required by the model
            sequence_data = latest_sequence_df[ENHANCED_FEATURES].values
            
            # Check for NaNs
            if not np.isfinite(sequence_data).all():
                logging.warning(f"Skipping {ticker}: Found NaN/Inf in latest feature data.")
                continue
                
            # Check shape
            if sequence_data.shape == (TIME_STEPS, len(ENHANCED_FEATURES)):
                results.append({
                    'ticker': ticker,
                    'sequence': sequence_data,
                    'last_date': latest_sequence_df.index[-1]
                })
            else:
                logging.warning(f"Skipping {ticker}: Incorrect data shape.")
        else:
            logging.debug(f"Skipping {ticker}: Not enough data ({len(stock_df)} rows).")

    if not results:
        logging.error("No valid sequences could be generated. Aborting.")
        return
        
    logging.info(f"Extracted {len(results)} valid sequences for prediction.")

    # --- 8. Prepare Data for LGBM ---
    logging.info("Scaling and flattening data for prediction...")
    tickers = [r['ticker'] for r in results]
    last_dates = [r['last_date'] for r in results]
    X_sequences = np.array([r['sequence'] for r in results], dtype=np.float32)

    n_samples, n_steps, n_features = X_sequences.shape
    X_flat = X_sequences.reshape(n_samples, n_steps * n_features)
    
    # Scale the data using the loaded scaler
    X_scaled = breakout_scaler.transform(X_flat)

    # --- 9. Run Prediction ---
    logging.info("Running LightGBM model predictions...")
    # Use the native booster 'predict' method
    pred_probabilities = breakout_model.predict(X_scaled)

    # --- 10. Format and Display Output ---
    logging.info("Generating confidence table...")
    
    if pred_probabilities.shape[1] != 3:
        logging.error(f"Model prediction shape is incorrect: {pred_probabilities.shape}. Expected 3 classes.")
        return

    confidence_df = pd.DataFrame({
        'Ticker': tickers,
        'Confidence_Loss (0)': pred_probabilities[:, 0],
        'Confidence_Timeout (1)': pred_probabilities[:, 1],
        'Confidence_Win (2)': pred_probabilities[:, 2],
        'Last_Data_Date': last_dates
    })
    
    confidence_df = confidence_df.sort_values('Confidence_Win (2)', ascending=False)
    
    # Save to CSV
    output_filename = "sp500_current_confidence.csv"
    try:
        confidence_df.to_csv(output_filename, index=False, float_format="%.4f")
        logging.info(f"Successfully saved full report to {output_filename}")
    except Exception as e:
        logging.error(f"Failed to save report to CSV: {e}")

    # Print top 25 to console
    print("\n--- S&P 500 Current Model Confidence (Top 25) ---")
    print(confidence_df.head(25).to_string(index=False, float_format="%.4f"))
    print("--------------------------------------------------")
    logging.info(f"--- Confidence Report Complete ---")


if __name__ == "__main__":
    try:
        generate_confidence_report()
    except Exception as e:
        logging.error(f"Unhandled exception in script: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logging.shutdown()