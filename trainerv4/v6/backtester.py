#!/usr/bin/env python3
"""
v6 - Backtester
Implements the walk-forward backtest logic.
Refined exit logic and added fixed stop-loss implementation.
"""

import logging
import pandas as pd
import numpy as np
from typing import Tuple # Added Tuple for type hint

# Assuming config is available for default BULLISH_REGIMES if not passed
# from .config import INITIAL_PORTFOLIO_CASH, BULLISH_REGIMES as DEFAULT_BULLISH_REGIMES
# Hardcoding default for simplicity if config isn't directly importable here
DEFAULT_BULLISH_REGIMES = [1, 2, 3] # Default based on last successful filter


def run_backtest_on_fold(
    predictions_df: pd.DataFrame,
    full_test_data: pd.DataFrame, # Expects Date index, includes OHLC, raw_atr, predicted_regime, ticker
    strategy_params: dict,
    initial_cash: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs a portfolio backtest on a single fold's OOS predictions.
    Uses either ATR trailing stop or fixed percentage stop.
    Applies regime filter if enabled.
    """
    start_date_pred = predictions_df['date'].min()
    end_date_pred = predictions_df['date'].max()
    logging.info(f"Running backtest from {start_date_pred} to {end_date_pred} with {initial_cash:.2f} starting cash")

    cash = float(initial_cash)
    portfolio = {} # Holds current positions: ticker -> {info}
    trades = [] # Log of all closed trades
    portfolio_values = [] # Daily snapshot of total portfolio value

    # --- Strategy Parameters ---
    cost_pct = strategy_params.get('transaction_cost_percent', 0.0005)
    slip_pct = strategy_params.get('slippage_percent', 0.0005)
    risk_per_trade_pct = strategy_params.get('risk_per_trade_percent', 0.01)
    holding_period = strategy_params.get('holding_period_days', 15)
    confidence_threshold = strategy_params.get('confidence_threshold', 0.60)
    use_regime_filter = strategy_params.get('use_regime_filter', False)
    bullish_regimes = strategy_params.get('BULLISH_REGIMES', DEFAULT_BULLISH_REGIMES)

    # --- Stop Loss Parameters ---
    use_trailing_stop = strategy_params.get('use_trailing_stop', False) # Default OFF based on analysis
    atr_stop_mult = strategy_params.get('atr_stop_multiplier', 4.0) # Looser default
    stop_loss_pct = strategy_params.get('stop_loss_percent', 0.06) # Fixed stop %

    # Prepare predictions lookup
    predictions_df['date'] = pd.to_datetime(predictions_df['date'])
    predictions_df = predictions_df.set_index('date').sort_index()
    unique_dates = predictions_df.index.unique().sort_values() # Ensure sorted dates

    # Ensure full_test_data has the correct index type (already set in main.py)
    if not isinstance(full_test_data.index, pd.DatetimeIndex):
         logging.error("Backtester requires full_test_data to have a DatetimeIndex.")
         return pd.DataFrame(), pd.DataFrame() # Return empty if data format is wrong


    # --- Main Backtest Loop ---
    for current_date in unique_dates:
        # --- 1. Mark-to-Market Portfolio Value ---
        current_portfolio_value = cash
        try:
            # Get all market data for the current date using index slice
            # This handles cases where only one row exists for the date
            todays_market_data = full_test_data.loc[full_test_data.index == current_date]
        except KeyError:
             # Handle cases where current_date might not be in market data (e.g., holiday)
             # Log value based on previous day's close if needed, or skip MTM for the day
             # For simplicity, we can use the last known portfolio value or carry forward prices
             if portfolio_values: # Use previous day's value if available
                 current_portfolio_value = portfolio_values[-1]['total_value']
             portfolio_values.append({'date': current_date, 'total_value': current_portfolio_value})
             continue # Skip trading logic if no market data

        if todays_market_data.empty:
             if portfolio_values: # Use previous day's value if available
                 current_portfolio_value = portfolio_values[-1]['total_value']
             portfolio_values.append({'date': current_date, 'total_value': current_portfolio_value})
             continue # Skip if market closed (e.g., weekend/holiday)


        # Update value of existing positions
        for ticker, info in portfolio.items():
            stock_data_today = todays_market_data[todays_market_data['ticker'] == ticker]
            if not stock_data_today.empty:
                market_price = float(stock_data_today['close'].iloc[0])
                current_portfolio_value += info['shares'] * market_price
            else:
                # If no data for stock today, use its last known price (entry price initially)
                # More sophisticated: could use previous day's close if available
                current_portfolio_value += info['shares'] * info.get('last_price', info['entry_price'])
                # Store the price used for next day's MTM if needed
                portfolio[ticker]['last_price'] = info.get('last_price', info['entry_price'])


        portfolio_values.append({'date': current_date, 'total_value': current_portfolio_value})


        # --- 2. Manage Existing Positions (Check Exits) ---
        for ticker, info in list(portfolio.items()):
            stock_data_today = todays_market_data[todays_market_data['ticker'] == ticker]
            if stock_data_today.empty:
                # Update last_price if needed, but can't check exits
                portfolio[ticker]['last_price'] = info.get('last_price', info['entry_price'])
                continue # No data for this stock today, hold

            # Extract today's prices
            row = stock_data_today.iloc[0] # Now it's a Series
            market_price = float(row['close'])
            market_high = float(row['high'])
            market_low = float(row['low'])
            portfolio[ticker]['last_price'] = market_price # Store today's close for next MTM

            exit_reason = None
            exit_price = market_price # Default exit price is close

            # --- Stop Loss Logic ---
            if use_trailing_stop:
                current_stop = info['stop_loss_price']
                current_atr = info['entry_atr'] # Use ATR from entry day
                new_potential_stop = market_high - (current_atr * atr_stop_mult)
                new_stop = max(current_stop, new_potential_stop)
                portfolio[ticker]['stop_loss_price'] = new_stop # Update stop

                if market_low <= new_stop:
                    exit_reason = 'trailing-stop'
                    # Exit at stop price, but not better than market low,
                    # and not worse than market high (in case of gap down)
                    exit_price = max(new_stop, market_low)
                    exit_price = min(exit_price, market_high) # Cap exit price at day's high
            else:
                 # --- NEW: Fixed Stop Loss Check ---
                 fixed_stop_price = info['entry_price_raw'] * (1 - stop_loss_pct) # Base stop on raw entry before slippage
                 if market_low <= fixed_stop_price:
                      exit_reason = 'fixed-stop'
                      # Exit at stop price, capped between low/high
                      exit_price = max(fixed_stop_price, market_low)
                      exit_price = min(exit_price, market_high)


            # --- Time Exit Check ---
            holding_days = (current_date - info['entry_date']).days
            if not exit_reason and holding_days >= holding_period:
                exit_reason = 'time-exit'
                exit_price = market_price # Exit at close for time exit

            # --- Process Exit ---
            if exit_reason:
                # Apply slippage and costs
                price_after_slippage = exit_price * (1 - slip_pct)
                proceeds = info['shares'] * price_after_slippage
                transaction_cost = proceeds * cost_pct
                cash += (proceeds - transaction_cost)

                profit = (proceeds - transaction_cost) - info['cost_basis']

                trades.append({
                    'exit_date': current_date,
                    'ticker': ticker,
                    'profit': profit,
                    'entry_regime': info['entry_regime'],
                    'exit_reason': exit_reason,
                    'entry_date': info['entry_date'],
                    'entry_price': info['entry_price'], # Price after slippage
                    'exit_price': exit_price, # Price before slippage/costs
                    'shares': info['shares'],
                    'win': 1 if profit > 0 else 0,
                    'holding_period': holding_days # Log actual holding period
                })
                del portfolio[ticker]

        # --- 3. Look for New Positions ---
        try:
            todays_preds = predictions_df.loc[current_date]
            # Ensure it's a DataFrame (in case of single prediction)
            if isinstance(todays_preds, pd.Series):
                todays_preds = todays_preds.to_frame().T
            todays_preds = todays_preds.sort_values('pred_win', ascending=False) # Rank by win probability
        except KeyError:
            continue # No predictions for today

        # Get entry data for predicted tickers
        predicted_tickers = todays_preds['ticker'].unique()
        entry_data_rows = todays_market_data[todays_market_data['ticker'].isin(predicted_tickers)]

        for _, signal in todays_preds.iterrows():
            ticker = signal['ticker']

            # --- Check if already holding ---
            if ticker in portfolio:
                continue

            # --- Get stock data for entry ---
            row_df = entry_data_rows[entry_data_rows['ticker'] == ticker]
            if row_df.empty:
                continue # No market data for this ticker today

            row = row_df.iloc[0] # Get the Series

            # --- Regime Filter Logic ---
            if use_regime_filter:
                if 'predicted_regime' in row.index and not pd.isna(row['predicted_regime']):
                    current_regime = int(row['predicted_regime'])
                    if current_regime not in bullish_regimes:
                        # logging.debug(f"{current_date.date()} Skipping {ticker}: Regime {current_regime} not bullish.")
                        continue # Skip trade, not a bullish regime
                else:
                    logging.warning(f"Ticker {ticker} missing 'predicted_regime' data on {current_date.date()}. Skipping regime check/trade.")
                    continue # Skip if regime unknown

            # --- Model Confidence Check (Still useful even if weak) ---
            win_probability = signal['pred_win']
            if win_probability < confidence_threshold:
                 # logging.debug(f"{current_date.date()} Skipping {ticker}: Confidence {win_probability:.2f} < {confidence_threshold}")
                 continue

            # --- Position Sizing ---
            price = float(row['close'])
            atr = float(row['raw_atr']) if 'raw_atr' in row.index and not pd.isna(row['raw_atr']) else price * 0.05 # Fallback
            if atr <= 0 or price <= 0:
                continue # Need valid price and ATR

            # Calculate stop distance based on chosen method
            if use_trailing_stop:
                dollar_risk_per_share = atr * atr_stop_mult
            else:
                dollar_risk_per_share = price * stop_loss_pct

            if dollar_risk_per_share <= 1e-6: # Avoid division by zero/tiny risk
                continue

            portfolio_risk_amount = current_portfolio_value * risk_per_trade_pct
            num_shares_to_buy = portfolio_risk_amount / dollar_risk_per_share

            # --- Calculate Costs & Check Cash ---
            price_after_slippage = price * (1 + slip_pct)
            amount_to_invest = num_shares_to_buy * price_after_slippage # Cost based on price after slippage
            transaction_cost = amount_to_invest * cost_pct
            total_cost = amount_to_invest + transaction_cost

            if cash >= total_cost:
                final_shares = num_shares_to_buy # Use the shares calc'd from risk
                cash -= total_cost

                # Determine initial stop price
                if use_trailing_stop:
                    initial_stop_price = price - (atr * atr_stop_mult) # Base initial stop on entry price
                else:
                    # For fixed stop, we don't need to store it dynamically, it's calculated on check
                    initial_stop_price = price * (1 - stop_loss_pct) # Store for reference if needed

                # Store position info
                portfolio[ticker] = {
                    'shares': final_shares,
                    'entry_price': price_after_slippage, # Store price including slippage
                    'entry_price_raw': price, # Store raw price for fixed stop calculation
                    'entry_date': current_date,
                    'entry_regime': row['predicted_regime'] if 'predicted_regime' in row.index else np.nan,
                    'cost_basis': total_cost,
                    'entry_atr': atr, # Store ATR at entry for trailing stop
                    'stop_loss_price': initial_stop_price, # Initial stop (will trail if use_trailing_stop=True)
                    'last_price': price_after_slippage # Initialize last known price
                }
                # logging.info(f"{current_date.date()} Entered {ticker} @ {price:.2f}, Shares: {final_shares:.0f}, Stop: {initial_stop_price:.2f}")


    # --- End Loop ---

    # Convert results to DataFrames
    portfolio_values_df = pd.DataFrame(portfolio_values).set_index('date').sort_index()
    trades_df_final = pd.DataFrame(trades)
    if not trades_df_final.empty:
         trades_df_final = trades_df_final.sort_values(by='exit_date')


    return portfolio_values_df, trades_df_final