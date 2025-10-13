import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
import pickle
from concurrent.futures import ThreadPoolExecutor
import logging
import requests

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Configuration ---
HISTORY_PERIOD = "5y"
TIME_STEPS = 60
BREAKOUT_PERIOD_DAYS = 15 # Needed for feature engineering consistency
CONFIDENCE_THRESHOLD = 0.60 # The AUC of ~0.61 suggests this is a reasonable starting point

# --- IMPORTANT: Use the filenames from your final training script ---
MODEL_FILENAME = 'breakout_tuned_model_wf_final.keras'
SCALER_FILENAME = 'scaler_wf_final.pkl'
METADATA_FILENAME = 'metadata_wf_final.pkl'

# --- Set up basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load the pre-trained model, scaler, and metadata ONCE at startup ---
try:
    logging.info(f"Loading model ({MODEL_FILENAME}), scaler ({SCALER_FILENAME}), and metadata...")
    model = tf.keras.models.load_model(MODEL_FILENAME)
    with open(SCALER_FILENAME, 'rb') as f:
        scaler = pickle.load(f)
    with open(METADATA_FILENAME, 'rb') as f:
        metadata = pickle.load(f)
    FEATURES = metadata['features'] # Load features from metadata
    logging.info("Model, scaler, and metadata loaded successfully.")
except Exception as e:
    logging.error(f"FATAL ERROR loading model artifacts: {e}")
    logging.error("Please ensure you have run the final training script to generate the required files.")
    model = None
    scaler = None
    FEATURES = []

# --- Feature Engineering Functions (IDENTICAL to final training script) ---
def get_stock_data(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    price_df = ticker.history(period=HISTORY_PERIOD, auto_adjust=True)
    return ticker_symbol, price_df if not price_df.empty else None

def process_data_for_prediction(df: pd.DataFrame, spy_df: pd.DataFrame, features_list: list) -> pd.DataFrame:
    """
    This function is now identical to the one in the final training script to prevent skew.
    It now also ensures the column order matches the training features.
    """
    if df is None: return None
    df = df.copy().sort_index()
    spy_df_renamed = spy_df.copy()[['Close']].rename(columns={'Close': 'SPY_Close'})
    df = df.merge(spy_df_renamed, left_index=True, right_index=True, how='left').ffill()

    close = df['Close']
    df['relative_strength'] = (df['Close'] / df['SPY_Close'])
    df['spy_sma_50_rel'] = (df['SPY_Close'] / df['SPY_Close'].rolling(50).mean()) - 1.0
    df['SMA_50_rel'] = (close / close.rolling(50).mean()) - 1.0
    df['ROC_20'] = close.pct_change(20)
    df['Volume_rel'] = (df['Volume'] / df['Volume'].rolling(20).mean()) - 1.0
    
    # --- FIX: Replaced pandas_ta bbands with the robust manual calculation from training ---
    bb_length = 20
    bb_std = 2.0
    rolling_mean = df['Close'].rolling(window=bb_length).mean()
    rolling_std = df['Close'].rolling(window=bb_length).std()
    upper_band = rolling_mean + (rolling_std * bb_std)
    lower_band = rolling_mean - (rolling_std * bb_std)
    df['BB_Width'] = (upper_band - lower_band) / rolling_mean
        
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
    
    # Drop rows with any NaN values created during feature calculation
    df.dropna(inplace=True)
    
    # Ensure all required feature columns exist, fill with 0 if one is missing
    for feature in features_list:
        if feature not in df.columns:
            df[feature] = 0
            logging.warning(f"Feature '{feature}' was missing. Filled with 0.")
            
    # Return dataframe with columns in the exact order the model expects
    return df[features_list]

# --- Core Logic Function: Analyze a single stock ---
def analyze_stock(ticker_data, spy_df):
    ticker, full_df = ticker_data
    logging.info(f"--- Analyzing {ticker} ---")
    if model is None or scaler is None or not FEATURES:
        return {"Ticker": ticker, "Signal": "Error: Model/Features not loaded"}

    try:
        if full_df is None: return {"Ticker": ticker, "Signal": "Data Error"}
        logging.info(f"[{ticker}] Initial data shape from yfinance: {full_df.shape}")

        processed_df = process_data_for_prediction(full_df, spy_df, FEATURES)
        logging.info(f"[{ticker}] Processed data shape after feature engineering & dropna: {processed_df.shape}")

        if processed_df.empty or len(processed_df) < TIME_STEPS:
            logging.warning(f"[{ticker}] Not enough data for model. Required: {TIME_STEPS}, Got: {len(processed_df)}")
            price = f"${full_df['Close'].iloc[-1]:.2f}" if not full_df.empty else "N/A"
            return {"Ticker": ticker, "Price": price, "Confidence": "", "Pred_Return": "", "Signal": "Not Enough Data"}

        last_sequence_raw = processed_df.tail(TIME_STEPS)
        last_sequence_scaled = scaler.transform(last_sequence_raw)
        last_sequence_reshaped = np.array([last_sequence_scaled])
        
        pred_bk, pred_ret = model.predict(last_sequence_reshaped, verbose=0)
        confidence = pred_bk[0][0]
        predicted_return = pred_ret[0][0]

        signal = "Potential Breakout" if confidence > CONFIDENCE_THRESHOLD else "Hold"
        
        return {
            "Ticker": ticker,
            "Confidence": f"{confidence:.2%}",
            "Pred_Return": f"{predicted_return:.2f}%",
            "Price": f"${full_df['Close'].iloc[-1]:.2f}",
            "Signal": signal
        }
    except Exception as e:
        logging.error(f"An error occurred while analyzing {ticker}: {e}")
        return {"Ticker": ticker, "Price": "N/A", "Confidence": "", "Pred_Return": "", "Signal": "Runtime Error"}

# --- Flask Web Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    initial_tickers = "AAPL, MSFT, NVDA, GOOGL, AMZN" # Fallback list

    if request.method == 'POST':
        tickers_string = request.form.get('tickers')
        initial_tickers = tickers_string
        tickers = [ticker.strip().upper() for ticker in tickers_string.split(',') if ticker.strip()]
        
        if tickers:
            _, spy_df = get_stock_data("SPY")
            if spy_df is None:
                logging.error("Could not fetch SPY data for analysis.")
                return render_template('index.html', results=[], initial_tickers=initial_tickers, error="Could not fetch market data (SPY).")

            with ThreadPoolExecutor(max_workers=10) as executor:
                ticker_data_list = list(executor.map(get_stock_data, tickers))
                results = list(executor.map(lambda td: analyze_stock(td, spy_df), ticker_data_list))
    
    return render_template('index.html', results=results, initial_tickers=initial_tickers)

@app.route('/get-most-active')
def get_most_active():
    """Fetches the most active stocks from Yahoo Finance."""
    try:
        #url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=most_actives"
        url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?count=200&formatted=true&scrIds=MOST_ACTIVES&sortField=&sortType=&start=0&useRecordsResponse=false&fields=symbol%2CshortName&lang=en-US&region=US"
        headers = {'User-Agent': 'Mozilla.5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        quotes = data.get('finance', {}).get('result', [{}])[0].get('quotes', [])
        if not quotes:
            return jsonify({"error": "Could not find tickers in API response."}), 500
        tickers = [quote['symbol'] for quote in quotes if 'symbol' in quote]
        return jsonify({"tickers": tickers})
    except Exception as e:
        logging.error(f"Error in get_most_active: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

