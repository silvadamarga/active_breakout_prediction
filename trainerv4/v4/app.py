import os
import json
import logging
import argparse
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from concurrent.futures import ThreadPoolExecutor

# --- Local Imports ---
from prediction_service import PredictionService
from data_fetcher import DataFetcher

# --- GPU Configuration Function ---
def setup_gpu_memory_growth():
    """Checks for GPUs and enables memory growth to prevent VRAM hogging."""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"TensorFlow memory growth enabled for {len(gpus)} GPU(s).")
        else:
            logging.info("No GPU found, running on CPU.")
    except RuntimeError as e:
        logging.error(f"Error setting memory growth: {e}")

# --- Initialize Flask App ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Instantiate Services ---
prediction_service = None
data_fetcher = None

# --- Configuration ---
TICKER_SOURCE_FILE = "news.json"
MARKET_SENTIMENT_FILE = "market_sentiment_db.json"

# --- Helper Functions ---
def load_news_data(filepath):
    """Loads news analysis data from a specific JSON file."""
    if not os.path.exists(filepath):
        logging.warning(f"News analysis file not found: {filepath}")
        return {}
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading or parsing {filepath}: {e}")
        return {}

def find_latest_news_file():

    """Finds the most recent 'news_analysis_YYYYMMDD.json' file."""
    base_name, ext = os.path.splitext(TICKER_SOURCE_FILE)
    try:
        # List all files in the current directory
        all_files = os.listdir('.')
        # Filter for files that match the pattern
        news_files = [f for f in all_files if f.startswith(base_name) and f.endswith(ext)]
        # Return the latest one if any exist
        if news_files:
            return sorted(news_files)[-1]
    except Exception as e:
        logging.error(f"Error searching for news files: {e}")
    return None

# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the main user interface."""
    model_load_error = "Critical Error: Model artifacts failed to load. The application is not functional." if not prediction_service else None
    return render_template('index.html', error=model_load_error)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handles the asynchronous analysis request from the frontend."""
    if not prediction_service or not data_fetcher:
        return jsonify({"error": "A backend service is not available. Please check server logs."}), 500

    data = request.get_json()
    tickers = data.get('tickers', [])
    
    # Always fetch regime tickers for the header status
    regime_tickers = ["SPY", "^VIX", "^TNX", "^FVX", "HYG", "IEF", "GLD"]
    all_tickers_to_fetch = list(set(tickers + regime_tickers))
    
    # If no tickers provided, just return the regime status for the initial page load
    if not tickers:
        price_data_map = data_fetcher.fetch_all_data(regime_tickers)
        regime_result = prediction_service.predict_current_regime(price_data_map)
        return jsonify({"results": [], "regime": regime_result})

    # --- Step 1: Fetch all required data ---
    price_data_map = data_fetcher.fetch_all_data(all_tickers_to_fetch)
    
    # --- Step 2: Predict the CURRENT Market Regime ---
    regime_result = prediction_service.predict_current_regime(price_data_map)
    predicted_regime = regime_result.get('regime_id') if isinstance(regime_result, dict) else None

    # --- Step 2.5: Calculate Risk Metrics ---
    risk_metrics = {}
    for ticker, df in price_data_map.items():
        if df is None or df.empty or ticker not in tickers: continue
        try:
            # Simple Stop-Loss: 20-day Simple Moving Average
            stop_loss = df['Close'].rolling(window=20).mean().iloc[-1]
            risk_metrics[ticker] = { "stop_loss": f"${stop_loss:.2f}" }
        except Exception as e:
            logging.warning(f"Could not calculate risk metrics for {ticker}: {e}")

    # --- Step 3: Prepare data for breakout prediction ---
    spy_df = price_data_map.get("SPY")
    if spy_df is None:
        return jsonify({"error": "Could not fetch SPY data. Analysis cannot proceed."}), 500
    
    breakout_price_map = {t: df for t, df in price_data_map.items() if t in tickers}
    prepared_data_list = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {
            executor.submit(prediction_service.prepare_data_for_prediction, ticker, df, spy_df, predicted_regime): ticker
            for ticker, df in breakout_price_map.items()
        }
        for future in future_to_ticker:
            try:
                prepared_data_list.append(future.result())
            except Exception as e:
                ticker = future_to_ticker[future]
                logging.error(f"Data preparation for {ticker} failed in thread: {e}", exc_info=True)
                prepared_data_list.append({"Ticker": ticker, "Signal_Raw": "Prep Failed", "Signal_Filtered": "Prep Failed"})
    
    # --- Step 4: Run batch prediction ---
    technical_results = prediction_service.run_batch_prediction(prepared_data_list, predicted_regime)
    
    # --- Step 5: Merge with News and Risk Data ---
    news_data = {}
    latest_news_file = "news.json"
    if latest_news_file:
        logging.info(f"Loading news from most recent file: {latest_news_file}")
        news_data = load_news_data(latest_news_file)
    else:
        logging.warning(f"No news analysis files found matching pattern: {TICKER_SOURCE_FILE}")

    final_results = []
    for result in technical_results:
        ticker = result['Ticker']
        
        # Merge news data
        if ticker_news := news_data.get(ticker):
            result['News_Positive'] = ticker_news.get('has_positive_news', False)
            result['News_Negative'] = ticker_news.get('has_negative_news', False)
            result['News_Confidence'] = f"{ticker_news.get('confidence', 0)}/10"
            result['News_Summary'] = ticker_news.get('summary', 'N/A')
        else:
            result['News_Positive'] = False
            result['News_Negative'] = False
            result['News_Confidence'] = 'N/A'
            result['News_Summary'] = 'N/A'
            
        # Merge risk metrics
        if ticker_metrics := risk_metrics.get(ticker):
            result.update(ticker_metrics)
            
        final_results.append(result)
        
    return jsonify({
        "results": sorted(final_results, key=lambda x: x.get('Ticker', '')),
        "regime": regime_result
    })

@app.route('/get-tickers-from-file')
def get_tickers_from_file():
    """Loads tickers from the most recent news analysis file."""
    latest_news_file = find_latest_news_file()
    if not latest_news_file:
        return jsonify({"error": f"No news analysis files found."}), 404
    try:
        with open(latest_news_file, 'r') as f: data = json.load(f)
        return jsonify({"tickers": list(data.keys())})
    except Exception as e:
        logging.error(f"Error reading or parsing {latest_news_file}: {e}")
        return jsonify({"error": f"Failed to process {latest_news_file}."}), 500

@app.route('/get-yahoo-most-active')
def get_yahoo_most_active():
    if not data_fetcher:
        return jsonify({"error": "Data fetching service is not available."}), 500
    try:
        tickers = data_fetcher.get_most_active()
        return jsonify({"tickers": tickers}) if tickers else jsonify({"error": "Failed to fetch most active tickers."}), 500
    except Exception as e:
        logging.error(f"Error in get_yahoo_most_active route: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

@app.route('/get-market-sentiment')
def get_market_sentiment():
    """Reads the last entry from the market sentiment DB."""
    if not os.path.exists(MARKET_SENTIMENT_FILE):
        logging.warning(f"Sentiment DB file not found: {MARKET_SENTIMENT_FILE}")
        return jsonify({"error": "Sentiment file not found."}), 404
    try:
        with open(MARKET_SENTIMENT_FILE, 'r') as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            return jsonify(data[-1]) # Return the last (most recent) entry
        else:
            logging.warning(f"Sentiment DB file '{MARKET_SENTIMENT_FILE}' is empty or not a list.")
            return jsonify({"error": "No sentiment data available."}), 404
    except Exception as e:
        logging.error(f"Error loading or parsing {MARKET_SENTIMENT_FILE}: {e}")
        return jsonify({"error": "Failed to process sentiment file."}), 500

def setup_arg_parser():
    parser = argparse.ArgumentParser(description="Stock Breakout Predictor Flask App")
    parser.add_argument('--gpu-mem-growth', action='store_true', help="Enable TensorFlow GPU memory growth.")
    return parser

if __name__ == '__main__':
    parser = setup_arg_parser()
    args = parser.parse_args()

    if args.gpu_mem_growth:
        setup_gpu_memory_growth()

    try:
        prediction_service = PredictionService()
        data_fetcher = DataFetcher()
        logging.info("All services initialized successfully.")
    except Exception as e:
        logging.critical(f"FATAL: A critical service failed to initialize: {e}", exc_info=True)

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)