import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
import yfinance as yf
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from flask import Flask, render_template, request

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Model & Backtesting Configuration (Slightly adjusted for screening) ---
HISTORY_PERIOD = "3y" 
TIME_STEPS = 60
BREAKOUT_PERIOD_DAYS = 15
BREAKOUT_THRESHOLD_PERCENT = 7.5
CONFIDENCE_THRESHOLD = 0.60 

# --- All Helper & Feature Engineering Functions (Copied from the backtester) ---
# These functions are self-contained and work perfectly here.
def get_stock_data(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    price_df = ticker.history(period=HISTORY_PERIOD, auto_adjust=False)
    return price_df if not price_df.empty else None

def process_data(price_df):
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
    future_returns = price_df['Close'].pct_change(BREAKOUT_PERIOD_DAYS).shift(-BREAKOUT_PERIOD_DAYS) * 100
    price_df['Target'] = (future_returns > BREAKOUT_THRESHOLD_PERCENT).astype(int)
    price_df.dropna(inplace=True)
    return price_df

def compute_rsi(series, period=14):
    delta = series.diff(1); gain = (delta.where(delta > 0, 0)).fillna(0).rolling(window=period).mean(); loss = (-delta.where(delta < 0, 0)).fillna(0).rolling(window=period).mean(); rs = gain / loss; rs.replace([np.inf, -np.inf], 0, inplace=True); rs.fillna(0, inplace=True); return 100 - (100 / (1 + rs))

def compute_atr(high, low, close, period=14):
    tr1 = high - low; tr2 = abs(high - close.shift()); tr3 = abs(low - close.shift()); tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1); return tr.ewm(alpha=1/period, adjust=False).mean()

def create_sequences(data, features, scaler):
    data_subset = data[features].copy(); scaled_data = scaler.fit_transform(data_subset); X, y = [], [];
    for i in range(TIME_STEPS, len(scaled_data)): X.append(scaled_data[i-TIME_STEPS:i]); y.append(data['Target'].iloc[i]);
    return np.array(X), np.array(y)

# --- Core Logic Function: Analyze a single stock ---
def analyze_stock(ticker):
    """
    This function encapsulates the entire analysis pipeline for one stock.
    It returns a dictionary with the results.
    """
    print(f"--- Analyzing {ticker} ---")
    try:
        full_df = get_stock_data(ticker)
        if full_df is None: return {"Ticker": ticker, "Score": "Data Error", "Confidence": "N/A", "Price": "N/A", "Signal": "Error"}
        
        processed_df = process_data(full_df)
        if processed_df is None or processed_df.empty: return {"Ticker": ticker, "Score": "Processing Error", "Confidence": "N/A", "Price": "N/A", "Signal": "Error"}

        # Model Training
        features = ['Close', 'SMA_50', 'RSI', 'MACD', 'ROC_20', 'Volume', 'Volume_Spike', 'ATR', 'Dist_from_52W_High']
        scaler = MinMaxScaler(feature_range=(0, 1))
        X, y = create_sequences(processed_df, features, scaler)

        if len(X) == 0 or np.unique(y).shape[0] < 2: return {"Ticker": ticker, "Score": "Model Error", "Confidence": "Not enough data", "Price": f"${full_df['Close'].iloc[-1]:.2f}", "Signal": "Hold"}
        
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = dict(enumerate(class_weights))

        model = Sequential([Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(X.shape[1], X.shape[2])), MaxPooling1D(pool_size=2), Dropout(0.2), LSTM(units=50, return_sequences=True, recurrent_dropout=0.1), Dropout(0.2), LSTM(units=50, recurrent_dropout=0.1), Dropout(0.2), Dense(units=1, activation='sigmoid')])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X, y, epochs=35, batch_size=32, verbose=0, class_weight=class_weight_dict)

        # Generate Prediction
        last_sequence_raw = processed_df[features].tail(TIME_STEPS)
        last_sequence_scaled = scaler.transform(last_sequence_raw)
        last_sequence_reshaped = np.array([last_sequence_scaled])
        confidence = model.predict(last_sequence_reshaped, verbose=0)[0][0]
        
        score = int(confidence * 100)
        signal = "Potential Breakout" if confidence > CONFIDENCE_THRESHOLD else "Hold"
        
        return {
            "Ticker": ticker,
            "Score": f"{score}/100",
            "Confidence": f"{confidence:.2%}",
            "Price": f"${full_df['Close'].iloc[-1]:.2f}",
            "Signal": signal
        }
    except Exception as e:
        print(f"An error occurred while analyzing {ticker}: {e}")
        return {"Ticker": ticker, "Score": "Runtime Error", "Confidence": "N/A", "Price": "N/A", "Signal": "Error"}


# --- Flask Web Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    if request.method == 'POST':
        tickers_string = request.form.get('tickers')
        # Sanitize and split tickers
        tickers = [ticker.strip().upper() for ticker in tickers_string.split(',') if ticker.strip()]
        
        for ticker in tickers:
            results.append(analyze_stock(ticker))
            
    return render_template('index.html', results=results)

if __name__ == '__main__':
    tf.random.set_seed(42)
    np.random.seed(42)
    # Important: host='0.0.0.0' makes the server accessible from outside the container
    app.run(host='0.0.0.0', port=5000, debug=True)
