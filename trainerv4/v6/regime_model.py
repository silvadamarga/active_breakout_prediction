#!/usr/bin/env python3
"""
v6 - Market Regime Model Module
Handles training, loading, and predicting market regimes.
"""

import os
import logging
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Dict, Tuple, Any
from sklearn.preprocessing import MinMaxScaler # Added missing import

from tensorflow.keras import Input, Model, optimizers
from tensorflow.keras.layers import (
    GRU, Dense, Dropout, Bidirectional, LayerNormalization, Masking
)
from tensorflow.keras.utils import to_categorical

# Import config and utils
from .config import (
    REGIME_TICKERS, REGIME_MODEL_FILENAME, REGIME_SCALER_FILENAME,
    REGIME_METADATA_FILENAME, HISTORY_PERIOD_YEARS, TIME_STEPS,
    REGIME_FUTURE_PERIOD_DAYS, BATCH_SIZE, REGIME_PREDS_CACHE
)
# DataManager needed for training/prediction logic
# Import locally in function to avoid circular deps if needed
# from .data_manager import DataManager

# ---------------- Regime Feature Engineering ----------------

def create_balanced_hybrid_regimes(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Creates balanced, hybrid economic regimes based on volatility and credit spread."""
    logging.info("Creating balanced, hybrid economic regimes...")
    # Calculate normalized volatility and credit spread
    norm_vol = (df['volatility_20d'] - df['volatility_20d'].rolling(252).mean()) / df['volatility_20d'].rolling(252).std()
    norm_spread = (df['credit_spread_ratio'] - df['credit_spread_ratio'].rolling(252).mean()) / df['credit_spread_ratio'].rolling(252).std()

    # Create composite risk score
    df['composite_risk'] = norm_vol + norm_spread

    # Initial regime based on quartiles of composite risk
    df['regime'] = pd.qcut(df['composite_risk'], 4, labels=False, duplicates='drop')

    # Override regime to 'Crisis' if yield curve is inverted
    df.loc[df['yield_curve_5y10y'] < 0, 'regime'] = 3 # Assign regime 3 (Crisis)

    regime_map = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk", 3: "Crisis"}
    logging.info(f"Regime distribution:\n{df['regime'].value_counts(normalize=True).sort_index()}")

    # Create future regime label (mode over next N days)
    df['future_regime'] = df['regime'].shift(-REGIME_FUTURE_PERIOD_DAYS).rolling(
        window=REGIME_FUTURE_PERIOD_DAYS, min_periods=1
    ).apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan, raw=False)

    return df, regime_map

def create_regime_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Calculates features specifically for the regime model."""
    logging.info("Calculating regime features...")
    # Price relative to Moving Averages
    df['SMA_50_rel'] = (df['Close'] / df['Close'].rolling(50).mean()) - 1.0
    df['SMA_200_rel'] = (df['Close'] / df['Close'].rolling(200).mean()) - 1.0

    # RSI and RSI Momentum
    rsi = df.ta.rsi(length=14)
    if isinstance(rsi, pd.Series): # Ensure it's a series before assigning
        df['RSI_14'] = rsi
        df['RSI_14_mom'] = df['RSI_14'].diff(5)
    else:
        logging.warning("Failed to calculate RSI for regime model.")
        df['RSI_14'] = np.nan
        df['RSI_14_mom'] = np.nan


    # Yield Curve Momentum
    df['yield_curve_mom'] = df['yield_curve_5y10y'].diff(5)

    # Relative performance vs. Bonds and Gold
    df['spy_vs_bonds'] = (df['Close'] / df.get('IEF_Close', df['Close'])).pct_change(20) # Use get for safety
    df['spy_vs_gold'] = (df['Close'] / df.get('GLD_Close', df['Close'])).pct_change(20)

    # Drop intermediate or raw columns before defining features
    df = df.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume',
                           'Dividends', 'Stock Splits', 'log_return',
                           'composite_risk'], errors='ignore')

    # Define the final feature list
    all_numeric_cols = list(df.select_dtypes(include=np.number).columns)
    exclude_cols = ['regime', 'future_regime'] + [col for col in df.columns if '_Close' in col] # Exclude raw prices
    FEATURES = [c for c in all_numeric_cols if c not in exclude_cols]

    logging.info(f"Regime features created: {FEATURES}")
    return df, FEATURES


# ---------------- Regime Model: Sequence Creation ----------------
def create_regime_sequences(df: pd.DataFrame, features: List[str]):
    """Creates sequences (X) and labels (y) for the regime model."""
    logging.info("Creating regime sequences...")
    # Fill NaNs *before* creating sequences
    data = df[features].ffill().fillna(0).values
    labels = df['future_regime'].values

    X, y, dates = [], [], []
    for i in range(TIME_STEPS, len(df)):
        # Check if the label for this timestep is valid
        if pd.notna(labels[i]):
            X.append(data[i-TIME_STEPS:i])
            y.append(int(labels[i])) # Ensure label is integer
            dates.append(df.index[i])

    if not X:
        logging.error("No valid regime sequences created. Check input data and feature calculation.")
        return np.array([]), np.array([]), np.array([]) # Return empty arrays

    logging.info(f"Created {len(X)} regime sequences.")
    return np.array(X), np.array(y), np.array(dates)


# ---------------- Regime Model: Architecture ----------------
def build_stacked_gru_model(n_features: int, n_classes: int):
    """Builds the stacked GRU model architecture for regime classification."""
    inp = Input(shape=(TIME_STEPS, n_features))
    x = Masking(mask_value=0.0)(inp) # Handle potential padding/missing steps if needed
    x = Bidirectional(GRU(units=64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, reset_after=False))(x)
    x = GRU(units=32, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, reset_after=False)(x)
    x = LayerNormalization()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(n_classes, activation='softmax', dtype='float32')(x) # Use float32 for softmax output
    model = Model(inputs=inp, outputs=output)

    optimizer = optimizers.AdamW(learning_rate=0.001, weight_decay=1e-5) # Use AdamW
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    logging.info("Regime model compiled.")
    return model


# ---------------- Regime Model: Training Orchestration ----------------
def train_or_load_regime_model(data_manager: Any) -> bool:
    """Trains a new regime model or confirms existing artifacts are present."""
    if all(os.path.exists(f) for f in [REGIME_MODEL_FILENAME, REGIME_SCALER_FILENAME, REGIME_METADATA_FILENAME]):
        logging.info("All regime model artifacts found. Loading existing model.")
        return True

    if data_manager.offline_mode:
        logging.error(f"Offline mode: Regime model artifacts not found. Cannot train. Aborting.")
        return False

    logging.warning("Regime model artifacts not found. Training new regime model...")
    try:
        # --- Import DataManager locally ---
        if 'DataManager' not in globals():
            from .data_manager import DataManager
        # --- End local import ---

        all_data = data_manager.fetch_all_data(REGIME_TICKERS)
        if any(t not in all_data for t in REGIME_TICKERS):
            logging.error("Failed to fetch critical market data for regime model. Halting."); return False

        # --- Data Preparation ---
        market_df = all_data["SPY"].copy()
        for t in REGIME_TICKERS:
            # Clean ticker name for column
            clean_name = t.replace('^','') + '_Close'
            if t != "SPY":
                 # Ensure join happens correctly even if SPY has fewer columns initially
                 if t in all_data:
                     market_df = market_df.join(all_data[t][['Close']].rename(columns={'Close': clean_name}), how='left')
                 else:
                     logging.warning(f"Data for {t} not found, column {clean_name} will be NaNs.")
                     market_df[clean_name] = np.nan


        # Calculate intermediate metrics needed for features/regimes
        market_df['log_return'] = np.log(market_df['Close'] / market_df['Close'].shift(1))
        market_df['volatility_20d'] = market_df['log_return'].rolling(20).std() * np.sqrt(252)
        market_df['yield_curve_5y10y'] = market_df.get('TNX_Close', np.nan) - market_df.get('FVX_Close', np.nan)
        market_df['credit_spread_ratio'] = market_df.get('HYG_Close', np.nan) / market_df.get('IEF_Close', np.nan)

        # Forward fill essential columns before feature/regime creation
        fill_cols = ['Close', 'volatility_20d', 'yield_curve_5y10y', 'credit_spread_ratio'] + [col for col in market_df if '_Close' in col]
        market_df[fill_cols] = market_df[fill_cols].ffill()
        market_df.bfill(inplace=True) # Backfill for initial NaNs

        # Create regimes and features
        market_df, regime_map = create_balanced_hybrid_regimes(market_df)
        market_df, FEATURES = create_regime_features(market_df)

        # Drop rows where future regime is unknown
        market_df.dropna(subset=['future_regime'], inplace=True)
        if market_df.empty:
            logging.error("No data remaining after dropping NaN future regimes. Aborting train.")
            return False

        # Create sequences
        X_all, y_all, _ = create_regime_sequences(market_df, FEATURES)
        if len(X_all) == 0:
             logging.error("Failed to create any sequences for regime model training. Aborting.")
             return False

        # Scale features
        # Reshape X for scaling, then reshape back
        scaler_final = MinMaxScaler().fit(X_all.reshape(-1, X_all.shape[-1]))
        X_all_s = scaler_final.transform(X_all.reshape(-1, X_all.shape[-1])).reshape(X_all.shape)

        # Convert labels to categorical
        y_all_cat = to_categorical(y_all, num_classes=len(regime_map))

        # Build and train the final model
        final_model = build_stacked_gru_model(len(FEATURES), len(regime_map))
        logging.info("Fitting final regime model...")
        final_model.fit(X_all_s, y_all_cat, epochs=30, batch_size=BATCH_SIZE*2, verbose=2) # Use larger batch for faster training

        # Save artifacts
        final_model.save(REGIME_MODEL_FILENAME)
        with open(REGIME_SCALER_FILENAME, 'wb') as f: pickle.dump(scaler_final, f)
        metadata = {'features': FEATURES, 'regime_map': regime_map, 'time_steps': TIME_STEPS}
        with open(REGIME_METADATA_FILENAME, 'wb') as f: pickle.dump(metadata, f)

        logging.info("Regime model training complete. Artifacts saved.")
        return True

    except Exception as e:
        logging.error(f"Failed to train regime model: {e}", exc_info=True)
        return False


# ---------------- Regime Model: Prediction Orchestration ----------------
def load_regime_model_and_artifacts():
    """Loads the pre-trained regime model and associated artifacts."""
    try:
        logging.info("Loading pre-trained market regime model and artifacts...")
        model = tf.keras.models.load_model(REGIME_MODEL_FILENAME, safe_mode=False) # Assume safe if needed
        with open(REGIME_SCALER_FILENAME, 'rb') as f: scaler = pickle.load(f)
        with open(REGIME_METADATA_FILENAME, 'rb') as f: metadata = pickle.load(f)
        logging.info("Regime model loaded successfully.")
        return model, scaler, metadata['features'], metadata['regime_map']
    except Exception as e:
        logging.error(f"FATAL: Could not load regime model at {REGIME_MODEL_FILENAME}. Error: {e}")
        return None, None, None, None

def predict_regimes_for_period(market_df: pd.DataFrame, model, scaler, features: list) -> pd.DataFrame:
    """Generates regime predictions for a given period using a trained model."""
    logging.info("Generating daily market regime predictions for the full period...")

    # Ensure necessary features are present and filled
    market_df_filled = market_df[features].ffill().fillna(0)
    data = market_df_filled.values

    X, dates = [], []
    for i in range(TIME_STEPS, len(market_df)):
        X.append(data[i-TIME_STEPS:i]); dates.append(market_df.index[i])

    if not X:
        logging.error("No sequences generated for regime prediction.")
        return pd.DataFrame()

    X = np.array(X)
    # Scale the sequences
    X_scaled = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    # Make predictions
    predictions_proba = model.predict(X_scaled, batch_size=BATCH_SIZE*4) # Larger batch for prediction
    predictions = np.argmax(predictions_proba, axis=1)

    return pd.DataFrame({'predicted_regime': predictions}, index=pd.to_datetime(dates))


# --- NEW ORCHESTRATOR FUNCTION ---
def build_and_predict_regime_df(
    data_manager: Any, # Use 'Any' to avoid import
    model: tf.keras.Model,
    scaler: MinMaxScaler,
    features: List[str],
    offline_mode: bool # Added missing parameter
) -> pd.DataFrame:
    """
    Orchestrates fetching data, calculating features, and predicting regimes.
    Handles caching of predictions.
    """
    if os.path.exists(REGIME_PREDS_CACHE):
        logging.info(f"Loading cached regime predictions from {REGIME_PREDS_CACHE}")
        try:
            return pd.read_pickle(REGIME_PREDS_CACHE)
        except Exception as e:
            logging.warning(f"Could not load regime cache: {e}. Regenerating...")

    if offline_mode:
        logging.error(f"Offline mode: Regime predictions cache not found at {REGIME_PREDS_CACHE}. Aborting.")
        return pd.DataFrame() # Return empty DataFrame on failure

    logging.info("Phase 2: Generating and caching regime predictions...")
    try:
        # --- Import DataManager locally ---
        if 'DataManager' not in globals():
            from .data_manager import DataManager
        # --- End local import ---

        regime_context_data = data_manager.fetch_all_data(REGIME_TICKERS)
        if any(t not in regime_context_data for t in REGIME_TICKERS):
             logging.error("Missing critical data for regime prediction. Aborting.")
             return pd.DataFrame()

        # --- Data Preparation (same as in training) ---
        market_df = regime_context_data["SPY"].copy()
        for t in REGIME_TICKERS:
            clean_name = t.replace('^','') + '_Close'
            if t != "SPY":
                 if t in regime_context_data:
                     market_df = market_df.join(regime_context_data[t][['Close']].rename(columns={'Close': clean_name}), how='left')
                 else: market_df[clean_name] = np.nan

        market_df['log_return'] = np.log(market_df['Close'] / market_df['Close'].shift(1))
        market_df['volatility_20d'] = market_df['log_return'].rolling(20).std() * np.sqrt(252)
        market_df['yield_curve_5y10y'] = market_df.get('TNX_Close', np.nan) - market_df.get('FVX_Close', np.nan)
        market_df['credit_spread_ratio'] = market_df.get('HYG_Close', np.nan) / market_df.get('IEF_Close', np.nan)

        fill_cols = ['Close', 'volatility_20d', 'yield_curve_5y10y', 'credit_spread_ratio'] + [col for col in market_df if '_Close' in col]
        market_df[fill_cols] = market_df[fill_cols].ffill()
        market_df.bfill(inplace=True)

        # Create *only* the features needed for prediction (no regime labels)
        market_df, _ = create_regime_features(market_df) # We don't need the returned feature list here
        market_df.fillna(0, inplace=True) # Fill any NaNs created during feature gen

        # Ensure all required features are present
        missing_features = [f for f in features if f not in market_df.columns]
        if missing_features:
            logging.error(f"Missing required regime features for prediction: {missing_features}. Aborting.")
            return pd.DataFrame()

        # Predict using the loaded model and scaler
        predicted_regimes_df = predict_regimes_for_period(market_df, model, scaler, features)

        if predicted_regimes_df.empty:
            logging.error("Failed to generate regime predictions (empty result). Aborting.")
            return pd.DataFrame()

        # Save to cache
        predicted_regimes_df.to_pickle(REGIME_PREDS_CACHE)
        logging.info(f"Saved regime predictions to {REGIME_PREDS_CACHE}")
        return predicted_regimes_df

    except Exception as e:
        logging.error(f"Error during regime prediction generation: {e}", exc_info=True)
        return pd.DataFrame()

