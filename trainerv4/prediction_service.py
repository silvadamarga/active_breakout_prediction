import lightgbm as lgb # Use LightGBM
import numpy as np
import pandas as pd
import pickle
import logging
import os
import tensorflow as tf # Added TF import
import pandas_ta as ta # Needed for feature calculation
from sklearn.preprocessing import MinMaxScaler # Import scaler

# --- Local Imports ---
from feature_engineering import create_regime_features_for_live

# --- Helper - Adapted from v6/breakout_processor ---
# (Includes necessary functions directly within this service for deployment simplicity)
def get_slope(array):
    """Calculates the slope of a 1D array using linear regression."""
    y = np.nan_to_num(array)
    valid_points = ~np.isnan(y)
    if valid_points.sum() < 2: return 0.0
    x = np.arange(len(y))
    try:
        slope = np.polyfit(x[valid_points], y[valid_points], 1)[0]
    except (np.linalg.LinAlgError, ValueError):
        slope = 0.0
    return slope if np.isfinite(slope) else 0.0

def _calculate_manual_bbands(close_series, length=20, std=2.0):
    """Manually calculates Bollinger Bands."""
    if close_series is None or close_series.empty or close_series.isna().all(): return None, None
    try:
        middle_band = ta.sma(close_series, length=length)
        stdev = ta.stdev(close_series, length=length)
        if middle_band is None or stdev is None or middle_band.isna().all() or stdev.isna().all(): return None, None
        upper_band = middle_band + (stdev * std)
        lower_band = middle_band - (stdev * std)
        return upper_band, lower_band
    except Exception: return None, None
# --- End Helper ---


class PredictionService:
    """
    Encapsulates the V6 LightGBM breakout model and the V4 regime model.
    Includes @tf.function fix for regime prediction.
    """
    def __init__(self,
                 # --- Updated Paths for LGBM/Enhanced Model ---
                 breakout_model_path=None,
                 breakout_scaler_path=None,
                 breakout_metadata_path=None,
                 # --- Regime paths remain the same ---
                 regime_model_path=None,
                 regime_scaler_path=None,
                 regime_metadata_path=None):
        """Loads all necessary artifacts for both models."""
        logging.info("Initializing PredictionService with V6 LGBM Breakout Model...")

        self.is_ready = False

        # Define artifact paths using environment variables or NEW defaults
        self.breakout_model_path = breakout_model_path or os.getenv('BREAKOUT_MODEL_PATH', 'final_tuned_model_v6_enhanced.txt')
        self.breakout_scaler_path = breakout_scaler_path or os.getenv('BREAKOUT_SCALER_PATH', 'final_scaler_v6_enhanced.pkl')
        self.breakout_metadata_path = breakout_metadata_path or os.getenv('BREAKOUT_METADATA_PATH', 'final_metadata_v6_enhanced.pkl')
        self.regime_model_path = regime_model_path or os.getenv('REGIME_MODEL_PATH', 'market_regime_model_v4.keras')
        self.regime_scaler_path = regime_scaler_path or os.getenv('REGIME_SCALER_PATH', 'market_regime_scaler_v4.pkl')
        self.regime_metadata_path = regime_metadata_path or os.getenv('REGIME_METADATA_PATH', 'market_regime_metadata_v4.pkl')

        try:
            # --- Load V6 LightGBM Breakout Model Artifacts ---
            logging.info(f"Loading breakout model from: {self.breakout_model_path}")
            self.lgbm_booster = lgb.Booster(model_file=self.breakout_model_path)
            logging.info(f"Loading breakout scaler from: {self.breakout_scaler_path}")
            with open(self.breakout_scaler_path, 'rb') as f: self.breakout_scaler = pickle.load(f)
            logging.info(f"Loading breakout metadata from: {self.breakout_metadata_path}")
            with open(self.breakout_metadata_path, 'rb') as f:
                breakout_meta = pickle.load(f)
                self.breakout_features = breakout_meta.get('original_features')
                self.breakout_time_steps = breakout_meta.get('time_steps', 60)
                self.trained_strategy_params = breakout_meta.get('strategy_params', {})
            if not self.breakout_features: raise ValueError("Could not load 'original_features' list from breakout metadata.")
            logging.info(f"Breakout model expects {len(self.breakout_features)} features over {self.breakout_time_steps} steps.")

            # --- Load V4 Regime Model Artifacts (Using TensorFlow) ---
            logging.info(f"Loading regime model from: {self.regime_model_path}")
            self.regime_model = tf.keras.models.load_model(self.regime_model_path, safe_mode=False) # Requires TF
            logging.info(f"Loading regime scaler from: {self.regime_scaler_path}")
            with open(self.regime_scaler_path, 'rb') as f: self.regime_scaler = pickle.load(f)
            logging.info(f"Loading regime metadata from: {self.regime_metadata_path}")
            with open(self.regime_metadata_path, 'rb') as f:
                regime_meta = pickle.load(f)
                self.regime_features = regime_meta['features']
                self.regime_map = regime_meta['regime_map']
                self.regime_time_steps = regime_meta.get('time_steps', 60)
            if self.breakout_time_steps != self.regime_time_steps:
                 logging.warning(f"Time steps mismatch: Breakout={self.breakout_time_steps}, Regime={self.regime_time_steps}. Using {self.breakout_time_steps} for breakout.")
            self.time_steps = self.breakout_time_steps # Primarily use breakout time steps

            # --- Configuration ---
            self.confidence_threshold = self.trained_strategy_params.get('confidence_threshold', 0.65)
            self.bullish_regimes = self.trained_strategy_params.get('BULLISH_REGIMES', [1, 2])
            self.breakout_period_days = self.trained_strategy_params.get('holding_period_days', 30)

            self.is_ready = True
            logging.info("PredictionService: All model artifacts loaded successfully. Service is ready.")
            logging.info(f"Using Confidence Threshold: {self.confidence_threshold}")
            logging.info(f"Using Bullish Regimes: {self.bullish_regimes}")

        except FileNotFoundError as e:
            logging.error(f"FATAL: Artifact not found: {e}. Service not ready.")
        except Exception as e:
            logging.error(f"FATAL: Error loading artifacts: {e}", exc_info=True)

    # --- V6 Feature Calculation (Adapted from breakout_processor) ---
    def _calculate_v6_features(self, ticker, df, context_dfs):
        """Calculates all ENHANCED_FEATURES for a single ticker DataFrame."""
        # (Feature calculation logic remains the same as previous full file)
        if df is None or df.empty: return None
        df = df.copy(); df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'},inplace=True,errors='ignore')
        required_cols = ['open','high','low','close','volume'];
        if not all(col in df.columns for col in required_cols): return None
        for col in required_cols: df[col] = pd.to_numeric(df[col], errors='coerce');
        df.dropna(subset=required_cols, inplace=True)
        if len(df) < 50: return None
        def safe_merge(df_left, context_key, cols_to_merge):
             context_df = context_dfs.get(context_key)
             if context_df is not None:
                existing_cols = [c for c in cols_to_merge if c in context_df.columns]
                if existing_cols:
                    if not isinstance(context_df.index, pd.DatetimeIndex): context_df.index = pd.to_datetime(context_df.index)
                    if not isinstance(df_left.index, pd.DatetimeIndex): df_left.index = pd.to_datetime(df_left.index)
                    return df_left.merge(context_df[existing_cols], left_index=True, right_index=True, how='left')
             return df_left
        df = safe_merge(df, 'SPY', ['SPY_Close']); df = safe_merge(df, 'VIX', ['VIX_Close','VIX_SMA_5','VIX_SMA_20'])
        df = safe_merge(df, 'TLT', ['TLT_Close']); df = safe_merge(df, 'breadth', ['pct_above_50d','pct_above_200d'])
        df.ffill(inplace=True); close = df['close']
        try:
            bbu, bbl = _calculate_manual_bbands(close);
            if bbu is not None: bbw = (bbu-bbl).replace([np.inf,-np.inf],np.nan); df['BBW_norm']=(bbw/close).replace([np.inf,-np.inf],np.nan); df['BBW_6mo_rank']=df['BBW_norm'].rolling(126,min_periods=30).rank(pct=True)
            else: df['BBW_norm'], df['BBW_6mo_rank'] = np.nan, np.nan
            df['raw_atr'] = ta.atr(df['high'],df['low'],close,length=14); df['ATR_norm']=(df['raw_atr']/close).replace([np.inf,-np.inf],np.nan); df['ATR_6mo_rank']=df['ATR_norm'].rolling(126,min_periods=30).rank(pct=True)
            df['ATR_norm_std_20d']=df['ATR_norm'].rolling(20,min_periods=10).std(); df['Choppiness_14d']=ta.chop(df['high'],df['low'],close,length=14)
            adx_df = ta.adx(df['high'],df['low'],close,length=14); df['ADX_14d']=adx_df['ADX_14'] if adx_df is not None and 'ADX_14' in adx_df.columns else np.nan
            sma_20=ta.sma(close,20); sma_50=ta.sma(close,50); df['SMA_50_slope']=sma_50.rolling(10,min_periods=5).apply(get_slope,raw=True).fillna(0) if sma_50 is not None else 0.0
            df['SMA_20_50_Ratio']=(sma_20/sma_50.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan) if sma_20 is not None and sma_50 is not None else np.nan
            df['ROC_20d'] = ta.roc(close, length=20)
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).clip(lower=0); obv = ta.obv(close, df['volume'])
            df['OBV_slope_20d'] = obv.rolling(20, min_periods=10).apply(get_slope, raw=True).fillna(0) if obv is not None else 0.0
            vol_sma_50 = ta.sma(df['volume'], 50); df['Volume_vs_Avg'] = (df['volume'] / vol_sma_50.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        except Exception as e: logging.error(f"Error TA features {ticker}: {e}", exc_info=False)
        def calculate_rs_slope(s1, s2_name, lookback=50, min_p=20):
            s2 = df.get(s2_name);
            if s1 is None or s2 is None or s1.empty or s2.empty: return pd.Series(0.0, index=df.index)
            s1a, s2a = s1.align(s2, join='left'); rs = (s1a / s2a.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
            return rs.rolling(lookback, min_periods=min_p).apply(get_slope, raw=True).fillna(0)
        df['RS_vs_SPX_slope'] = calculate_rs_slope(close, 'SPY_Close'); df['RS_vs_Sector_slope'] = calculate_rs_slope(close, 'Sector_Close'); df['RS_vs_Bonds_slope'] = calculate_rs_slope(close, 'TLT_Close')
        df['VIX_level'] = df.get('VIX_Close'); vix_sma_20 = df.get('VIX_SMA_20'); df['VIX_MA_ratio'] = (df.get('VIX_SMA_5') / vix_sma_20.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan) if vix_sma_20 is not None else np.nan
        # Add regime passed from prediction
        # (This function doesn't add 'predicted_regime', it's added externally before calling predict)
        for feat in self.breakout_features:
            if feat not in df.columns: df[feat] = 0.0 # Add placeholder if missing
        df.fillna(0.0, inplace=True) # Final fill
        # Ensure only the expected features are returned in the correct order
        return df[self.breakout_features]


    # --- Internal decorated function for Keras regime prediction ---
    @tf.function # Add the decorator
    def _run_regime_prediction(self, sequence_scaled):
        """Internal function to run Keras predict within tf.function context."""
        # Call the model directly for compatibility with tf.function
        return self.regime_model(sequence_scaled, training=False)


    # --- Regime Prediction (Uses the decorated function) ---
    def predict_current_regime(self, market_data: dict) -> dict:
        """Predicts the current market regime using the V4 Keras model."""
        if not self.is_ready: return {"error": "Regime model not available."}
        try:
            market_df = create_regime_features_for_live(market_data) # Uses feature_engineering.py
            if market_df is None or len(market_df) < self.regime_time_steps:
                return {"error": "Not enough data/features for regime prediction."}

            # Select features required by the *regime* model
            sequence_data = market_df[self.regime_features].tail(self.regime_time_steps).values
            sequence_scaled = self.regime_scaler.transform(
                sequence_data.reshape(-1, len(self.regime_features))
            ).reshape(1, self.regime_time_steps, -1) # Reshape for Keras RNN

            # --- Use the decorated function ---
            prediction_tensor = self._run_regime_prediction(tf.constant(sequence_scaled, dtype=tf.float32)) # Ensure input is tensor
            prediction = prediction_tensor.numpy() # Convert tensor result back to numpy array
            # --- End Change ---

            if prediction is None or prediction.shape[0] == 0:
                 raise ValueError("Regime model prediction returned an empty result.")

            regime_id = np.argmax(prediction[0])
            regime_name = self.regime_map.get(str(regime_id), self.regime_map.get(regime_id, f"Unknown({regime_id})"))

            return {
                "regime_id": int(regime_id),
                "regime_name": regime_name,
                "is_bullish": int(regime_id) in self.bullish_regimes,
                "probabilities": prediction[0].tolist()
            }
        except Exception as e:
            logging.error(f"Regime prediction failed: {e}", exc_info=True)
            tf_error_msg = "`tf.data.Dataset` only supports" # Check specific error
            if tf_error_msg in str(e): return {"error": "TF context issue."}
            else: return {"error": f"Regime prediction internal error."}

    # --- Data Prep for V6 LGBM Model ---
    def prepare_data_for_prediction(self, ticker, price_df, context_dfs):
        """Prepares a single stock's data using V6 features for LGBM prediction."""
        if not self.is_ready:
             return {"Ticker": ticker, "Price": "N/A", "Signal_Raw": "Service Unavailable"}

        last_price_str = "N/A"
        if price_df is not None and not price_df.empty and 'Close' in price_df.columns:
            try: last_price_str = f"${price_df['Close'].iloc[-1]:.2f}"
            except IndexError: pass # Handle empty df after check?

        try:
            # Add predicted regime to context before calculating features
            # (Assuming regime prediction happens once before prepping all tickers)
            # This requires regime prediction result to be passed into context_dfs
            # If regime is added *during* feature calculation, adjust _calculate_v6_features

            features_df = self._calculate_v6_features(ticker, price_df, context_dfs)

            if features_df is None or len(features_df) < self.time_steps:
                logging.debug(f"{ticker}: Not enough data after feature calc ({len(features_df) if features_df is not None else 0} < {self.time_steps})")
                return {"Ticker": ticker, "Price": last_price_str, "Signal_Raw": "Not Enough Data"}

            # Get sequence and flatten
            sequence = features_df.tail(self.time_steps).values
            sequence_flat = sequence.reshape(1, -1)

            # Scale flattened sequence
            sequence_scaled = self.breakout_scaler.transform(sequence_flat)

            return {"type": "data", "Ticker": ticker, "Price": last_price_str, "sequence": sequence_scaled[0]} # Return 1D

        except Exception as e:
            logging.error(f"V6 Data prep failed for {ticker}: {e}", exc_info=True)
            return {"Ticker": ticker, "Price": last_price_str, "Signal_Raw": "Prep Error"}

    # --- Batch Prediction for V6 LGBM Model ---
    def run_batch_prediction(self, prepared_data_list, current_regime: int | None):
        """Runs batch prediction using LightGBM and returns signals."""
        if not self.is_ready:
             return [{"Ticker": d.get("Ticker","?"), "Price": d.get("Price","N/A"), "Signal_Raw": "Service Unavailable", "Signal_Filtered": "Service Unavailable"}
                     for d in prepared_data_list]

        valid_data = [d for d in prepared_data_list if d.get("type") == "data" and "sequence" in d]
        error_results = [d for d in prepared_data_list if d.get("type") != "data" or "sequence" not in d]
        for err in error_results: # Ensure error results have standard keys
             err.setdefault('Signal_Raw', 'Prep Failed'); err.setdefault('Signal_Filtered', err['Signal_Raw'])
             err.setdefault('Confidence', 'N/A')

        if not valid_data: return error_results

        sequences_batch = np.array([d['sequence'] for d in valid_data])

        try:
            # Predict probabilities using LightGBM booster
            pred_probabilities = self.lgbm_booster.predict(sequences_batch) # Shape: (n_samples, 3)
        except Exception as pred_e:
            logging.error(f"LightGBM prediction failed: {pred_e}", exc_info=True)
            # Return error state for all valid data entries if prediction itself fails
            for d in valid_data:
                error_results.append({"Ticker": d["Ticker"], "Price": d["Price"], "Signal_Raw": "Predict Error", "Signal_Filtered": "Predict Error", "Confidence": "N/A"})
            return error_results


        prediction_results = []
        for i, data in enumerate(valid_data):
            try:
                probs = pred_probabilities[i] # Probabilities for [Loss, Timeout, Win] (Indices 0, 1, 2)
                win_probability = probs[2] # Probability of "Win" class

                is_breakout_signal = win_probability > self.confidence_threshold
                signal_raw = "Potential Breakout" if is_breakout_signal else "Hold"
                signal_filtered = signal_raw

                # Apply regime filter
                if current_regime is not None and is_breakout_signal and (current_regime not in self.bullish_regimes):
                    signal_filtered = "Hold (Regime Filter)"

                prediction_results.append({
                    "Ticker": data['Ticker'], "Price": data['Price'],
                    "Confidence": f"{win_probability:.1%}",
                    "Signal_Raw": signal_raw, "Signal_Filtered": signal_filtered,
                })
            except IndexError:
                 logging.error(f"Prediction result index error for {data['Ticker']}. Probs shape: {pred_probabilities.shape}")
                 prediction_results.append({"Ticker": data["Ticker"], "Price": data["Price"], "Signal_Raw": "Predict Index Error", "Signal_Filtered": "Predict Index Error", "Confidence": "N/A"})
            except Exception as loop_e:
                 logging.error(f"Error processing prediction loop for {data['Ticker']}: {loop_e}")
                 prediction_results.append({"Ticker": data["Ticker"], "Price": data["Price"], "Signal_Raw": "Result Proc Error", "Signal_Filtered": "Result Proc Error", "Confidence": "N/A"})


        return prediction_results + error_results