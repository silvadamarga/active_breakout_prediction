import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
import time
import yfinance as yf
import os
import matplotlib.pyplot as plt

# --- TensorFlow and Keras Imports ---
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout

# --- Backtesting Configuration ---
STOCK_TICKERS = ['TSM', 'MSFT', 'SNOW', 'ASTS', 'COIN', 'TTWO', 'INTC', 'BIDU', 'AMD', 'PLTR', 'RGTI' ] 
HISTORY_PERIOD = "5y" 
TRAIN_TEST_SPLIT_RATIO = 0.8 
INITIAL_CASH = 10000.0
HOLDING_PERIOD_DAYS = 15

# --- Model Configuration ---
TIME_STEPS = 60
BREAKOUT_PERIOD_DAYS = 15
BREAKOUT_THRESHOLD_PERCENT = 7.5
CONFIDENCE_THRESHOLD = 0.60 

# --- Helper and Feature Engineering Functions (Unchanged) ---
def get_stock_data(ticker_symbol):
    print(f"Fetching 5-year data for {ticker_symbol}...")
    ticker = yf.Ticker(ticker_symbol)
    price_df = ticker.history(period=HISTORY_PERIOD, auto_adjust=False)
    if price_df.empty: return None
    return price_df

def process_data(price_df):
    if price_df is None or price_df.empty or len(price_df) < 252: return None
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

# --- Model Training (UPDATED with Class Weights) ---
def train_hybrid_model(df):
    features = ['Close', 'SMA_50', 'RSI', 'MACD', 'ROC_20', 'Volume', 'Volume_Spike', 'ATR', 'Dist_from_52W_High']; scaler = MinMaxScaler(feature_range=(0, 1)); X, y = create_sequences(df, features, scaler)
    if len(X) == 0 or np.unique(y).shape[0] < 2: print("Not enough data or target variance to train."); return None, None, None
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- NEW: Calculate and apply class weights ---
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Calculated class weights: {class_weight_dict}") # To see how imbalanced it is

    model = Sequential([Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])), MaxPooling1D(pool_size=2), Dropout(0.2), LSTM(units=50, return_sequences=True, recurrent_dropout=0.1), Dropout(0.2), LSTM(units=50, recurrent_dropout=0.1), Dropout(0.2), Dense(units=1, activation='sigmoid')]); model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']);
    print("Training Hybrid model with class weights to focus on breakouts..."); 
    # Pass the class weights to the fit method
    model.fit(X_train, y_train, epochs=35, batch_size=32, validation_split=0.1, verbose=0, class_weight=class_weight_dict)
    
    return model, scaler, features

def run_backtest(model, scaler, features, test_data):
    print("Running backtest simulation...")
    cash = INITIAL_CASH; position = 0; days_in_trade = 0; portfolio_values = []; trades = []
    for i in range(TIME_STEPS, len(test_data)):
        current_sequence_raw = test_data[features].iloc[i-TIME_STEPS:i]; current_sequence_scaled = scaler.transform(current_sequence_raw); current_sequence_reshaped = np.array([current_sequence_scaled]);
        confidence = model.predict(current_sequence_reshaped, verbose=0)[0][0]; current_price = test_data['Close'].iloc[i]
        if position > 0:
            days_in_trade += 1
            if days_in_trade >= HOLDING_PERIOD_DAYS: cash += position * current_price; trades.append({'date': test_data.index[i], 'action': 'sell', 'price': current_price, 'shares': position, 'profit': (position * current_price) - trades[-1]['cost']}); position = 0; days_in_trade = 0
        elif confidence > CONFIDENCE_THRESHOLD:
            shares_to_buy = cash / current_price; position = shares_to_buy; cash = 0; trades.append({'date': test_data.index[i], 'action': 'buy', 'price': current_price, 'shares': shares_to_buy, 'cost': shares_to_buy * current_price})
        portfolio_values.append(cash + (position * current_price))
    return portfolio_values, trades

def analyze_performance(portfolio_values, trades, test_data, ticker):
    if not portfolio_values: print(f"No trades were made for {ticker}."); return
    final_value = portfolio_values[-1]; total_return = (final_value / INITIAL_CASH - 1) * 100; buy_and_hold_return = (test_data['Close'].iloc[-1] / test_data['Close'].iloc[0] - 1) * 100;
    portfolio_series = pd.Series(portfolio_values, index=test_data.index[TIME_STEPS:]); peak = portfolio_series.expanding(min_periods=1).max(); drawdown = (portfolio_series - peak) / peak; max_drawdown = drawdown.min() * 100
    wins = sum(1 for t in trades if t['action'] == 'sell' and t['profit'] > 0); total_trades = len([t for t in trades if t['action'] == 'sell']); win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    print("\n--- Backtest Results ---"); print(f"Total Return: {total_return:.2f}%"); print(f"Buy & Hold Return: {buy_and_hold_return:.2f}%"); print(f"Max Drawdown: {max_drawdown:.2f}%"); print(f"Total Trades: {total_trades}"); print(f"Win Rate: {win_rate:.2f}%");
    plt.figure(figsize=(15, 7)); plt.plot(portfolio_series.index, portfolio_series.values, label='Our Strategy', color='blue'); buy_and_hold_values = (test_data['Close'][TIME_STEPS:] / test_data['Close'].iloc[TIME_STEPS]) * INITIAL_CASH; plt.plot(buy_and_hold_values.index, buy_and_hold_values.values, label='Buy and Hold', color='gray', linestyle='--')
    buy_signals = [t['date'] for t in trades if t['action'] == 'buy']; sell_signals = [t['date'] for t in trades if t['action'] == 'sell']; plt.scatter(buy_signals, test_data.loc[buy_signals]['Close'], marker='^', color='green', s=100, label='Buy Signal', zorder=5); plt.scatter(sell_signals, test_data.loc[sell_signals]['Close'], marker='v', color='red', s=100, label='Sell Signal', zorder=5)
    plt.title(f'Backtest Performance for {ticker} (Balanced Model)'); plt.xlabel('Date'); plt.ylabel('Portfolio Value ($)'); plt.legend(); plt.grid(True)
    if not os.path.exists('backtests'): os.makedirs('backtests'); plt.savefig(f'backtests/{ticker}_backtest_balanced.png'); print(f"Backtest chart saved to backtests/{ticker}_backtest_balanced.png"); plt.close()

if __name__ == "__main__":
    print("\n--- AI Stock Backtester (Balanced) Initializing ---\n"); tf.random.set_seed(42); np.random.seed(42)
    for ticker in STOCK_TICKERS:
        print(f"\n===== Starting Backtest for {ticker} =====\n"); full_df = get_stock_data(ticker)
        if full_df is not None:
            processed_df = process_data(full_df)
            if processed_df is not None:
                split_index = int(len(processed_df) * TRAIN_TEST_SPLIT_RATIO); train_df = processed_df.iloc[:split_index]; test_df = processed_df.iloc[split_index:]
                model, scaler, features = train_hybrid_model(train_df)
                if model: portfolio_values, trades = run_backtest(model, scaler, features, test_df); analyze_performance(portfolio_values, trades, test_df, ticker)
        print("-" * 55); time.sleep(1)
