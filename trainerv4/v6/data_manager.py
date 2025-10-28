#!/usr/bin/env python3
"""
v6 - Data Manager
Handles fetching and caching of all market data (SQLite, yfinance).
Also includes data sync logic (S&P500 tickers, sector maps).
Ensures TLT is fetched for intermarket analysis.
"""

import os
import logging
import pickle
import json
import sqlite3
import time
from datetime import datetime
from io import StringIO
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Any # Added Any

import pandas as pd
import requests
import yfinance as yf

# Import configuration constants
from .config import (
    DB_CACHE_FILE, HISTORY_PERIOD_YEARS, SP500_TICKERS_CACHE,
    SECTOR_MAP_CACHE, TICKERS_JSON_FILE, REGIME_TICKERS, SECTOR_ETF_MAP,
    CACHE_DIR # Import CACHE_DIR for SP500/Sector map paths
)

class DataManager:
    """Manages fetching and caching of market data using SQLite and yfinance."""
    def __init__(self, db_path: str, offline_mode: bool = False):
        self.db_path = db_path
        self.offline_mode = offline_mode
        # Ensure directory for SQLite DB exists
        try:
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        except OSError as e:
            logging.error(f"Could not create directory for DB cache at {os.path.dirname(db_path)}: {e}")
            # Decide how to handle this - maybe raise error or proceed if DB exists?
        self._create_table()

    def _create_table(self):
        """Creates the stock_data table if it doesn't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS stock_data (
                        ticker TEXT PRIMARY KEY,
                        fetch_date TEXT,
                        data BLOB
                    )
                """)
        except sqlite3.Error as e:
            logging.error(f"Failed to create/connect to SQLite table: {e}")
            # Consider raising exception if DB is essential

    def get_stock_data(self, ticker: str) -> Tuple[str, pd.DataFrame | None]:
        """
        Retrieves stock data for a ticker, first from cache, then yfinance.
        Returns (ticker, DataFrame | None).
        """
        # 1. Try fetching from SQLite cache
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT data FROM stock_data WHERE ticker=?", (ticker,))
                result = cursor.fetchone()
                if result and result[0]:
                    # Attempt to unpickle data
                    try:
                        df = pickle.loads(result[0])
                        if isinstance(df, pd.DataFrame):
                            return ticker, df
                        else:
                            logging.warning(f"Cached data for {ticker} is not a DataFrame. Refetching.")
                    except pickle.UnpicklingError as pe:
                        logging.warning(f"Could not unpickle cached data for {ticker}: {pe}. Refetching.")
                    except Exception as e_inner:
                        logging.warning(f"Unexpected error loading cached data for {ticker}: {e_inner}. Refetching.")
        except sqlite3.Error as e_sql:
            logging.warning(f"SQLite error reading {ticker} from cache: {e_sql}")
        except Exception as e_outer:
             logging.warning(f"Unexpected error accessing cache for {ticker}: {e_outer}")

        # 2. Handle offline mode if cache miss
        if self.offline_mode:
            logging.warning(f"Offline mode: Cache miss or invalid cache for {ticker}. Returning None.")
            return ticker, None

        # 3. Fetch from yfinance if online and cache miss/invalid
        logging.info(f"Online mode: Cache miss/invalid for {ticker}, fetching from yfinance...")
        max_retries = 3
        backoff_factor = 2
        for i in range(max_retries):
            try:
                # Use yf.download for potential robustness or Ticker for more info access
                # df = yf.download(ticker, period=f"{HISTORY_PERIOD_YEARS}y", auto_adjust=True, progress=False)
                yf_ticker = yf.Ticker(ticker)
                df = yf_ticker.history(period=f"{HISTORY_PERIOD_YEARS}y", auto_adjust=True)

                if df.empty:
                    logging.warning(f"No data for {ticker} from yfinance.")
                    # Cache the fact that no data was found? Maybe not, could be temporary.
                    return ticker, None

                df.index = df.index.tz_localize(None) # Remove timezone info
                df = df.rename_axis('Date') # Ensure index is named 'Date'

                # Cache the fetched data
                try:
                    with sqlite3.connect(self.db_path) as conn:
                        # Serialize DataFrame using pickle
                        serialized_df = pickle.dumps(df)
                        conn.execute("INSERT OR REPLACE INTO stock_data (ticker, fetch_date, data) VALUES (?, ?, ?)",
                                     (ticker, datetime.now().strftime('%Y-%m-%d'), serialized_df))
                    logging.debug(f"Successfully cached data for {ticker}")
                    return ticker, df
                except sqlite3.Error as e_sql_write:
                    logging.error(f"SQLite error writing {ticker} to cache: {e_sql_write}")
                    return ticker, df # Return data even if caching failed
                except pickle.PicklingError as pe_write:
                     logging.error(f"Could not pickle data for {ticker}: {pe_write}")
                     return ticker, df # Return data even if serialization failed

            except Exception as e_fetch:
                wait_time = backoff_factor ** i
                logging.warning(f"Error fetching {ticker} (attempt {i+1}/{max_retries}): {e_fetch}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

        logging.error(f"Failed to fetch {ticker} after {max_retries} retries.")
        return ticker, None

    def fetch_all_data(self, tickers: List[str], max_workers: int = 10) -> Dict[str, pd.DataFrame]:
        """Fetches data for multiple tickers concurrently using ThreadPoolExecutor."""
        if not tickers:
            return {}
        logging.info(f"DataManager fetching/loading {len(tickers)} tickers (max_workers={max_workers})...")
        valid_data = {}
        # Use ThreadPoolExecutor for concurrent fetching
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # map preserves order and is simpler if individual error handling isn't critical
            results = list(executor.map(self.get_stock_data, tickers))

        for ticker, df in results:
            if df is not None and not df.empty:
                valid_data[ticker] = df

        successful_count = len(valid_data)
        failed_count = len(tickers) - successful_count
        logging.info(f"{('Offline load' if self.offline_mode else 'Online fetch/load')} complete. "
                     f"Successfully retrieved data for {successful_count}/{len(tickers)} tickers. "
                     f"{failed_count} failures or no data.")
        return valid_data

# ---------------- Data Sync Phase Functions ----------------

def get_sp500_tickers_cached(offline_mode: bool = False) -> list:
    """Gets S&P 500 tickers, from cache or by scraping Wikipedia."""
    cache_path = SP500_TICKERS_CACHE # Use path from config
    if os.path.exists(cache_path):
        logging.info(f"Loading S&P 500 tickers from local cache: {cache_path}")
        try:
            with open(cache_path, 'r') as f:
                tickers = json.load(f)
                if isinstance(tickers, list): return tickers
                else: logging.error(f"S&P 500 cache file {cache_path} is not a list.")
        except json.JSONDecodeError as je:
            logging.error(f"Error decoding S&P 500 cache file {cache_path}: {je}")
        except Exception as e:
            logging.error(f"Error reading S&P 500 cache file {cache_path}: {e}")
        # If cache read failed and offline, return empty
        if offline_mode: return []

    if offline_mode:
        logging.error(f"Offline mode: S&P 500 ticker cache not found or invalid at {cache_path}. Aborting.")
        return []

    # --- Online Scraping ---
    logging.info("Online mode: Scraping S&P 500 tickers from Wikipedia...")
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; MyQuantBot/1.0; +http://mybot.com/info)'} # More descriptive UA
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status() # Check for HTTP errors

        tables = pd.read_html(StringIO(r.text))
        if not tables:
            logging.error("No tables found on Wikipedia S&P 500 page.")
            return []
        # Find the correct table (usually the first one with 'Symbol' column)
        sp500_table = None
        for table in tables:
            if 'Symbol' in table.columns:
                sp500_table = table
                break
        if sp500_table is None:
            logging.error("Could not find table with 'Symbol' column on Wikipedia page.")
            return []

        tickers = sp500_table['Symbol'].tolist()
        # Clean tickers: replace '.' with '-', remove whitespace
        tickers = [str(t).replace('.', '-').strip() for t in tickers if isinstance(t, str) and t]
        tickers = sorted(list(set(tickers))) # Deduplicate and sort

        logging.info(f"Scraped {len(tickers)} S&P 500 tickers. Caching to {cache_path}")
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(tickers, f, indent=4) # Add indent for readability
        except Exception as e_write:
             logging.error(f"Could not write S&P 500 cache to {cache_path}: {e_write}")
        return tickers

    except requests.exceptions.RequestException as e_req:
        logging.error(f"Network error scraping S&P500 tickers: {e_req}")
    except Exception as e_scrape:
        logging.error(f"Error scraping or processing S&P500 tickers: {e_scrape}")
    return [] # Return empty list on failure

def get_tickers_from_json(filename: str) -> list:
    """Loads a list of custom tickers from a JSON file (list or dict keys)."""
    if not os.path.exists(filename):
        logging.warning(f"Custom ticker file '{filename}' not found. Skipping.")
        return []
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict):
            tickers = list(data.keys())
        elif isinstance(data, list):
            tickers = data
        else:
            raise ValueError("JSON file format must be a list of tickers or a dictionary.")
        # Clean tickers
        tickers = [str(t).strip().upper() for t in tickers if isinstance(t, str) and t] # Standardize to upper
        tickers = sorted(list(set(tickers))) # Deduplicate and sort
        logging.info(f"Loaded {len(tickers)} unique custom tickers from '{filename}'.")
        return tickers
    except json.JSONDecodeError as je:
         logging.error(f"Error decoding JSON from '{filename}': {je}")
    except Exception as e:
        logging.error(f"Could not read or parse custom ticker file '{filename}': {e}")
    return []

def get_or_create_sector_map(tickers: List[str], offline_mode: bool = False) -> dict:
    """Gets ticker-to-sector mapping, from cache or yfinance."""
    cache_path = SECTOR_MAP_CACHE # Use path from config
    if os.path.exists(cache_path):
        logging.info(f"Loading ticker-to-sector map from local cache: {cache_path}")
        try:
            with open(cache_path, 'r') as f:
                sector_map = json.load(f)
                if isinstance(sector_map, dict): return sector_map
                else: logging.error(f"Sector map cache file {cache_path} is not a dictionary.")
        except json.JSONDecodeError as je:
             logging.error(f"Error decoding sector map cache {cache_path}: {je}")
        except Exception as e:
            logging.error(f"Error reading sector map cache {cache_path}: {e}")
        if offline_mode: return {}

    if offline_mode:
        logging.error(f"Offline mode: Sector map cache not found or invalid at {cache_path}. Aborting.")
        return {}

    # --- Online Fetching ---
    logging.info(f"Online mode: Fetching sector info for {len(tickers)} tickers (using yfinance bulk)...")
    ticker_to_sector = {}

    # Use yf.Tickers for potentially faster bulk info fetching
    # Split into chunks if there are too many tickers for one request
    chunk_size = 100
    ticker_chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    processed_count = 0

    for chunk in ticker_chunks:
        try:
             # Ensure tickers are passed as a space-separated string or list to yf.Tickers
             ticker_list_str = " ".join(chunk)
             tickers_yf = yf.Tickers(ticker_list_str)

             # Iterate through the original chunk list to access results
             for ticker_str in chunk:
                 try:
                     # Access info using the upper-case ticker string
                     # yfinance Tickers object stores results keyed by UPPERCASE ticker
                     info = tickers_yf.tickers[ticker_str.upper()].info
                     sec = info.get('sector')
                     # Validate sector info
                     if sec and isinstance(sec, str) and sec != 'N/A':
                         ticker_to_sector[ticker_str] = sec # Use original ticker case as key
                     else:
                          logging.debug(f"No valid sector found for {ticker_str}. Info: {info.get('sector', 'Not Found')}")
                 except KeyError:
                     logging.debug(f"Ticker {ticker_str.upper()} not found in yf.Tickers result batch.")
                 except AttributeError:
                      # This can happen if yf.Tickers didn't create an entry for a ticker
                      logging.debug(f"Could not access info object for {ticker_str} in bulk fetch.")
                 except Exception as e_inner:
                      # Catch other errors for individual tickers within the chunk
                      logging.debug(f"Could not get info for {ticker_str}: {e_inner}")

             processed_count += len(chunk)
             logging.info(f"Fetched sector batch... {processed_count}/{len(tickers)}")
             time.sleep(0.5) # Small sleep between chunks to be polite

        except Exception as e_bulk:
             logging.warning(f"Error during bulk yfinance info fetch for chunk: {e_bulk}. Individual results may be missing.")
             processed_count += len(chunk) # Assume processed even on error for progress counting


    successful_fetches = len(ticker_to_sector)
    logging.info(f"Fetched sector info for {successful_fetches}/{len(tickers)} tickers. Caching to {cache_path}")

    # Cache the results
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(ticker_to_sector, f, indent=4)
    except Exception as e_write:
        logging.error(f"Could not write sector map cache to {cache_path}: {e_write}")

    return ticker_to_sector

# --- Orchestrator Function ---
def run_data_sync_phase(data_manager: DataManager, offline_mode: bool = False) -> Tuple[list, dict]:
    """
    Handles fetching tickers, sector map, and caching all required historical data.
    Ensures SPY, VIX, TLT, sector ETFs, regime tickers, and base tickers are fetched.
    """
    logging.info(f"--- Starting Data Sync Phase (Offline Mode: {offline_mode}) ---")

    # 1. Get Base Tickers
    tickers_sp = get_sp500_tickers_cached(offline_mode)
    custom_tickers = get_tickers_from_json(TICKERS_JSON_FILE)
    base_tickers = sorted(list(set(tickers_sp + custom_tickers)))
    if not base_tickers:
        logging.error("No base tickers found (S&P500 and custom file failed?). Aborting data sync.")
        return [], {}
    logging.info(f"Identified {len(base_tickers)} unique base tickers.")

    # 2. Get Sector Map
    ticker_to_sector = get_or_create_sector_map(base_tickers, offline_mode)
    if not ticker_to_sector and offline_mode:
         logging.error("Offline mode: Could not load required sector map. Aborting data sync.")
         return [], {}
    elif not ticker_to_sector:
        logging.warning("Could not load or fetch sector map. Sector features will be compromised.")

    # 3. Identify All Tickers to Fetch
    sector_etfs = list(set(SECTOR_ETF_MAP.values()))
    all_tickers_to_fetch = list(set(
        base_tickers + sector_etfs + REGIME_TICKERS + ['^VIX', 'TLT'] # Added 'TLT'
    ))
    all_tickers_to_fetch = sorted([t for t in all_tickers_to_fetch if t]) # Clean and sort
    logging.info(f"Identified {len(all_tickers_to_fetch)} unique tickers required for fetching.")

    # 4. Fetch/Cache All Data
    data_manager.fetch_all_data(all_tickers_to_fetch)

    logging.info("--- Data Sync Phase Complete ---")
    return base_tickers, ticker_to_sector