import pandas as pd
import numpy as np
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
HISTORY_PERIOD = "3y"
TIME_STEPS = 60
BREAKOUT_PERIOD_DAYS = 15 # Needed for feature engineering consistency
BREAKOUT_THRESHOLD_PERCENT = 7.5 # Needed for feature engineering consistency
CONFIDENCE_THRESHOLD = 0.60
MODEL_FILENAME = 'breakout_model.keras'
SCALER_FILENAME = 'scaler.pkl'

# --- Set up basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load the pre-trained model and scaler ONCE at startup ---
try:
    logging.info("Loading pre-trained model and scaler...")
    model = tf.keras.models.load_model(MODEL_FILENAME)
    with open(SCALER_FILENAME, 'rb') as f:
        scaler = pickle.load(f)
    logging.info("Model and scaler loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model or scaler: {e}")
    logging.error("Please ensure you have run train.py to generate the required files.")
    model = None
    scaler = None

# --- Helper & Feature Engineering Functions (Identical to training script) ---
def get_stock_data(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    price_df = ticker.history(period=HISTORY_PERIOD, auto_adjust=False)
    return price_df if not price_df.empty else None

def compute_rsi(series, period=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0).fillna(0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).fillna(0).rolling(window=period).mean()
    rs = gain / loss
    rs.replace([np.inf, -np.inf], 0, inplace=True)
    rs.fillna(0, inplace=True)
    return 100 - (100 / (1 + rs))

def compute_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def process_data_for_prediction(price_df):
    if price_df is None or len(price_df) < 252: return None
    price_df['SMA_50'] = price_df['Close'].rolling(window=50).mean()
    price_df['RSI'] = compute_rsi(price_df['Close'])
    exp12 = price_df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = price_df['Close'].ewm(span=26, adjust=False).mean()
    price_df['MACD'] = exp12 - exp26
    price_df['ROC_20'] = price_df['Close'].pct_change(20) * 100
    price_df['Volume_SMA_20'] = price_df['Volume'].rolling(window=20).mean()
    price_df['Volume_Spike'] = (price_df['Volume'] > price_df['Volume_SMA_20'] * 1.5).astype(int)
    price_df['ATR'] = compute_atr(price_df['High'], price_df['Low'], price_df['Close'])
    rolling_max = price_df['Close'].rolling(window=252).max()
    price_df['Dist_from_52W_High'] = (price_df['Close'] - rolling_max) / rolling_max * 100
    price_df.dropna(inplace=True)
    return price_df

# --- Core Logic Function: Analyze a single stock with the pre-trained model ---
def analyze_stock(ticker):
    logging.info(f"--- Analyzing {ticker} ---")
    if model is None or scaler is None:
        return {"Ticker": ticker, "Confidence": "N/A", "Price": "N/A", "Signal": "Error: Model not loaded"}

    try:
        full_df = get_stock_data(ticker)
        if full_df is None: return {"Ticker": ticker, "Confidence": "N/A", "Price": "N/A", "Signal": "Data Error"}

        processed_df = process_data_for_prediction(full_df)
        if processed_df is None or len(processed_df) < TIME_STEPS:
            return {"Ticker": ticker, "Confidence": "N/A", "Price": f"${full_df['Close'].iloc[-1]:.2f}", "Signal": "Not Enough Data"}

        features = ['Close', 'SMA_50', 'RSI', 'MACD', 'ROC_20', 'Volume', 'Volume_Spike', 'ATR', 'Dist_from_52W_High']
        last_sequence_raw = processed_df[features].tail(TIME_STEPS)
        last_sequence_scaled = scaler.transform(last_sequence_raw)
        last_sequence_reshaped = np.array([last_sequence_scaled])
        confidence = model.predict(last_sequence_reshaped, verbose=0)[0][0]
        signal = "Potential Breakout" if confidence > CONFIDENCE_THRESHOLD else "Hold"
        return {
            "Ticker": ticker,
            "Confidence": f"{confidence:.2%}",
            "Price": f"${full_df['Close'].iloc[-1]:.2f}",
            "Signal": signal
        }
    except Exception as e:
        logging.error(f"An error occurred while analyzing {ticker}: {e}")
        return {"Ticker": ticker, "Confidence": "N/A", "Price": "N/A", "Signal": "Runtime Error"}

# --- Flask Web Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    initial_tickers = ""

    if request.method == 'POST':
        tickers_string = request.form.get('tickers')
        initial_tickers = tickers_string # Keep the submitted tickers in the box
        tickers = [ticker.strip().upper() for ticker in tickers_string.split(',') if ticker.strip()]
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(analyze_stock, tickers))
    else: # On GET request, pre-fill with most active
        try:
            # This logic is similar to get_most_active but tailored for pre-filling the form
            url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=most_actives"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            quotes = data.get('finance', {}).get('result', [{}])[0].get('quotes', [])
            if quotes:
                tickers = [quote['symbol'] for quote in quotes if 'symbol' in quote]
                initial_tickers = ", ".join(tickers)
        except Exception as e:
            logging.error(f"Could not pre-fetch most active stocks: {e}")
            initial_tickers = "AAPL, MSFT, NVDA, GOOGL, AMZN" # Fallback list
            
    return render_template('index.html', results=results, initial_tickers=initial_tickers)

@app.route('/get-most-active')
def get_most_active():
    """
    Fetches the most active stocks from the Yahoo Finance screener API.
    This is called by the 'Most Active' button on the frontend.
    """
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=most_actives"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        quotes = data.get('finance', {}).get('result', [{}])[0].get('quotes', [])
        
        if not quotes:
            return jsonify({"error": "Could not find tickers in the API response. The API structure may have changed."}), 500

        tickers = [quote['symbol'] for quote in quotes if 'symbol' in quote]
        
        return jsonify({"tickers": tickers})

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching most active stocks from API: {e}")
        return jsonify({"error": "Failed to connect to Yahoo Finance API."}), 503
    except (KeyError, IndexError, TypeError) as e:
        logging.error(f"Error parsing API response for most active stocks: {e}")
        return jsonify({"error": "Failed to parse the API data from Yahoo Finance. Structure may have changed."}), 500
    except Exception as e:
        logging.error(f"An unexpected error occurred in get_most_active: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

