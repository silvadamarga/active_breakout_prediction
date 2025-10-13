import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import tensorflow as tf
import pickle
from concurrent.futures import ThreadPoolExecutor
import logging
import requests
from datetime import datetime

# --- Configuration (Mirrors app.py) ---
HISTORY_PERIOD = "5y"
TIME_STEPS = 60
CONFIDENCE_THRESHOLD = 0.60
MODEL_FILENAME = 'breakout_tuned_model_wf_final.keras'
SCALER_FILENAME = 'scaler_wf_final.pkl'
METADATA_FILENAME = 'metadata_wf_final.pkl'
OUTPUT_FILENAME = 'RESULTS.md'

# --- Set up basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Feature Engineering & Analysis Functions (Copied from app.py) ---

def get_stock_data(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    price_df = ticker.history(period=HISTORY_PERIOD, auto_adjust=True)
    return ticker_symbol, price_df if not price_df.empty else None

def get_most_active_stocks():
    """Fetches the most active stocks from Yahoo Finance."""
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=most_actives"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        quotes = data.get('finance', {}).get('result', [{}])[0].get('quotes', [])
        if not quotes:
            logging.error("Could not find tickers in API response.")
            return []
        tickers = [quote['symbol'] for quote in quotes if 'symbol' in quote]
        logging.info(f"Successfully fetched {len(tickers)} most active stocks.")
        return tickers
    except Exception as e:
        logging.error(f"Error fetching most active stocks: {e}")
        return []

def process_data_for_prediction(df: pd.DataFrame, spy_df: pd.DataFrame, features_list: list) -> pd.DataFrame:
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
    
    df.dropna(inplace=True)
    
    for feature in features_list:
        if feature not in df.columns:
            df[feature] = 0
    return df[features_list]

def analyze_stock(ticker_data, spy_df, model, scaler, features):
    ticker, full_df = ticker_data
    if full_df is None: return {"Ticker": ticker, "Signal": "Data Error"}

    processed_df = process_data_for_prediction(full_df, spy_df, features)
    
    if processed_df.empty or len(processed_df) < TIME_STEPS:
        price = f"${full_df['Close'].iloc[-1]:.2f}" if not full_df.empty else "N/A"
        return {"Ticker": ticker, "Price": price, "Confidence": "N/A", "Pred_Return": "N/A", "Signal": "Not Enough Data"}

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

def main():
    """Main function to run the analysis and save results."""
    logging.info("Starting analysis run...")
    
    # Load model and artifacts
    try:
        model = tf.keras.models.load_model(MODEL_FILENAME)
        with open(SCALER_FILENAME, 'rb') as f: scaler = pickle.load(f)
        with open(METADATA_FILENAME, 'rb') as f: metadata = pickle.load(f)
        features = metadata['features']
    except Exception as e:
        logging.error(f"Could not load model artifacts. Aborting. Error: {e}")
        return

    # Get tickers to analyze
    tickers = get_most_active_stocks()
    if not tickers:
        logging.warning("No tickers to analyze. Exiting.")
        return

    # Fetch data
    _, spy_df = get_stock_data("SPY")
    if spy_df is None:
        logging.error("Could not fetch SPY data. Aborting.")
        return
        
    with ThreadPoolExecutor(max_workers=10) as executor:
        ticker_data_list = list(executor.map(get_stock_data, tickers))
        results = list(executor.map(lambda td: analyze_stock(td, spy_df, model, scaler, features), ticker_data_list))

    # Filter for potential breakouts and sort by confidence
    breakouts = sorted(
        [res for res in results if res['Signal'] == 'Potential Breakout'],
        key=lambda x: float(x['Confidence'].strip('%')),
        reverse=True
    )
    
    # Write results to a markdown file
    with open(OUTPUT_FILENAME, 'w') as f:
        f.write(f"# Stock Breakout Analysis\n\n")
        f.write(f"**Last Updated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
        
        if not breakouts:
            f.write("No potential breakouts found matching the criteria.\n")
        else:
            f.write("| Ticker | Price | Confidence | Pred. Return |\n")
            f.write("|--------|-------|------------|--------------|\n")
            for res in breakouts:
                f.write(f"| {res['Ticker']} | {res['Price']} | {res['Confidence']} | {res['Pred_Return']} |\n")
                
    logging.info(f"Analysis complete. Results saved to {OUTPUT_FILENAME}")

if __name__ == '__main__':
    main()
