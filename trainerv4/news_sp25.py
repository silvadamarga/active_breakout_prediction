#!/usr/bin/env python3
"""
v7.1 - Stock Analyzer with Split DB Storage (Refactored & Fixed)

Fetches news and yfinance data for top N stocks (from confidence CSV),
analyzes them individually using Gemini for scores and summaries,
separates static/dynamic data, and saves results to a two-table SQLite database.
Optionally performs market sentiment analysis.

Fixes:
- Removed duplicate function definitions.
- Fixed broken 'main' loop and 'NameError' for 'all_db_results' & 'yf_static_data'.
- Implemented functional news hash caching with a '--force-analysis' flag.
- Cleaned up variable initializations and constants.
"""

import os
import re
import time
import json
import feedparser
import html
import logging
import argparse
from datetime import datetime, timedelta, timezone, date
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import google.generativeai as genai
from dotenv import load_dotenv
import pandas as pd
import yfinance as yf
import sqlite3
import sys
import math

# --- Load Environment Variables ---
load_dotenv()

# --- Script Dependencies ---
# pip install google-generativeai feedparser requests python-dotenv pandas yfinance

# --- Configuration ---
class Config:
    """Holds all script configurations, loaded from environment variables."""
    GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
    API_CALL_DELAY_SECONDS = int(os.getenv("API_CALL_DELAY_SECONDS", 2))
    NEWS_FETCH_DELAY_SECONDS = float(os.getenv("NEWS_FETCH_DELAY_SECONDS", 0.1))
    YFINANCE_FETCH_DELAY_SECONDS = float(os.getenv("YFINANCE_FETCH_DELAY_SECONDS", 0.1))
    MAX_NEWS_ARTICLES_PER_TICKER = int(os.getenv("MAX_NEWS_ARTICLES_PER_TICKER", 5))
    NEWS_CACHE_FILENAME = os.getenv("NEWS_CACHE_FILENAME", "news_cache.json")
    NEWS_TIME_HOURS = int(os.getenv("NEWS_TIME_HOURS", 24))
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    MARKET_SENTIMENT_DB = os.getenv("MARKET_SENTIMENT_DB", "market_sentiment_db.json")
    CONFIDENCE_COLUMN_NAME = 'Confidence_Win (2)'
    TICKER_COLUMN_NAME = 'Ticker'
    ANALYSIS_DB_FILENAME = os.getenv("ANALYSIS_DB_FILENAME", "stock_analysis_v2.db") # Updated DB name

# --- Global Constants ---
YF_STATIC_KEYS = ['longName', 'sector', 'industry', 'longBusinessSummary']

DEFAULT_ANALYSIS_ERROR_STATE = {
    "reasoning": "Error",
    "bullishness_score": 1,
    "news_sentiment_score": 1,
    "growth_score": 1,
    "valuation_score": 1,
    "financial_health_score": 1,
    "positive_summary": "N/A",
    "negative_summary": "N/A"
}

# --- Logging and Global Session ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_robust_session():
    """Creates a requests.Session with retries and a user-agent."""
    session = requests.Session()
    session.headers.update({'User-Agent': Config.USER_AGENT})
    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        backoff_factor=1
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

http_session = create_robust_session()

def initialize_gemini():
    """Configures and returns the Gemini generative model."""
    try:
        if not Config.GEMINI_API_KEY:
            logging.warning("GEMINI_API_KEY not set. Gemini features will fail.")
            return None
        genai.configure(api_key=Config.GEMINI_API_KEY)
        safety_settings=[
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]
        model = genai.GenerativeModel(Config.GEMINI_MODEL_NAME, safety_settings=safety_settings)
        logging.info(f"Gemini API configured: model {Config.GEMINI_MODEL_NAME}.")
        return model
    except Exception as e:
        logging.error(f"FATAL configuring Gemini API: {e}")
        return None

# --- Database Functions ---
def initialize_database():
    """Creates the 'companies' (static) and 'daily_analysis' (dynamic) tables."""
    try:
        with sqlite3.connect(Config.ANALYSIS_DB_FILENAME) as conn:
            cursor = conn.cursor()
            # Create companies table (Static Info)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS companies (
                    ticker TEXT PRIMARY KEY,
                    company_name TEXT,
                    sector TEXT,
                    industry TEXT,
                    business_summary TEXT,
                    last_static_update TEXT
                )
            """)
            # Create daily_analysis table (Dynamic Info & Scores)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_analysis (
                    ticker TEXT NOT NULL,
                    analysis_date TEXT NOT NULL,
                    model_confidence_win REAL,
                    bullishness_score INTEGER,
                    reasoning TEXT,
                    news_sentiment_score INTEGER,
                    growth_score INTEGER,
                    valuation_score INTEGER,
                    financial_health_score INTEGER,
                    positive_summary TEXT,
                    negative_summary TEXT,
                    headlines TEXT, -- JSON string
                    news_hash TEXT,
                    yf_data_json TEXT, -- JSON string of DYNAMIC yf data used
                    last_updated TEXT NOT NULL,
                    PRIMARY KEY (ticker, analysis_date),
                    FOREIGN KEY (ticker) REFERENCES companies(ticker)
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_ticker_date ON daily_analysis (ticker, analysis_date)")
            conn.commit()
            logging.info(f"Database {Config.ANALYSIS_DB_FILENAME} initialized with split tables.")
    except sqlite3.Error as e:
        logging.error(f"Error initializing split DB: {e}")
        raise

def save_results_to_db(analysis_data_list):
    """Saves analysis results to the split database tables."""
    if not analysis_data_list:
        logging.info("No analysis data provided to save.")
        return

    company_sql_ignore = """
        INSERT OR IGNORE INTO companies (
            ticker, company_name, sector, industry, business_summary, last_static_update
        ) VALUES (?, ?, ?, ?, ?, ?)
    """
    daily_sql = """
        INSERT OR REPLACE INTO daily_analysis (
            ticker, analysis_date, model_confidence_win, bullishness_score, reasoning,
            news_sentiment_score, growth_score, valuation_score, financial_health_score,
            positive_summary, negative_summary, headlines, news_hash, yf_data_json, last_updated
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    company_rows = []
    daily_rows = []
    current_timestamp = datetime.now(timezone.utc).isoformat()
    analysis_date = date.today().isoformat()
    seen_tickers = set() # To avoid duplicate company inserts in one run

    for data in analysis_data_list:
        ticker = data.get('ticker')
        if not ticker:
            continue

        # Prepare company data only once per ticker in this run
        if ticker not in seen_tickers:
            static_data = data.get('yf_static_data', {})
            company_rows.append((
                ticker,
                static_data.get('longName'),
                static_data.get('sector'),
                static_data.get('industry'),
                static_data.get('longBusinessSummary'),
                current_timestamp # Timestamp of this update attempt
            ))
            seen_tickers.add(ticker)

        # Prepare daily data
        daily_rows.append((
            ticker, analysis_date, data.get('model_confidence_win'),
            data.get('bullishness_score'), data.get('reasoning'), data.get('news_sentiment_score'),
            data.get('growth_score'), data.get('valuation_score'), data.get('financial_health_score'),
            data.get('positive_summary'), data.get('negative_summary'), json.dumps(data.get('headlines', [])),
            data.get('news_hash'), json.dumps(data.get('yf_dynamic_data', {})), # Save dynamic data used
            current_timestamp
        ))

    try:
        with sqlite3.connect(Config.ANALYSIS_DB_FILENAME) as conn:
            cursor = conn.cursor()
            if company_rows:
                cursor.executemany(company_sql_ignore, company_rows)
            if daily_rows:
                cursor.executemany(daily_sql, daily_rows)
            conn.commit()
            logging.info(f"DB Update: Processed {len(seen_tickers)} companies, saved/updated {len(daily_rows)} daily analyses.")
    except sqlite3.Error as e:
        logging.error(f"Error saving results to split DB: {e}")

# --- Data Fetching and Parsing Functions ---

def read_top_tickers_from_csv(filepath, top_n):
    """Reads CSV, returns dict {ticker: confidence_score} for top N."""
    logging.info(f"Reading top {top_n} tickers from: {filepath}")
    ticker_confidence_map = {}
    try:
        df = pd.read_csv(filepath)
        if Config.TICKER_COLUMN_NAME not in df.columns or Config.CONFIDENCE_COLUMN_NAME not in df.columns:
            logging.error(f"CSV needs '{Config.TICKER_COLUMN_NAME}' and '{Config.CONFIDENCE_COLUMN_NAME}' cols.")
            return {}
        
        df[Config.CONFIDENCE_COLUMN_NAME] = pd.to_numeric(df[Config.CONFIDENCE_COLUMN_NAME], errors='coerce')
        df = df.dropna(subset=[Config.CONFIDENCE_COLUMN_NAME, Config.TICKER_COLUMN_NAME])
        df = df[df[Config.TICKER_COLUMN_NAME].apply(lambda x: isinstance(x, str))] # Ensure ticker is string

        df_sorted = df.sort_values(by=Config.CONFIDENCE_COLUMN_NAME, ascending=False).head(top_n)

        for _, row in df_sorted.iterrows():
            ticker = str(row[Config.TICKER_COLUMN_NAME]).strip().upper()
            confidence = row[Config.CONFIDENCE_COLUMN_NAME]
            if ticker and isinstance(confidence, (int, float)) and not math.isnan(confidence):
                ticker_confidence_map[ticker] = confidence

        logging.info(f"Selected {len(ticker_confidence_map)} valid tickers with confidence scores.")
        return ticker_confidence_map
    except FileNotFoundError:
        logging.error(f"Input CSV '{filepath}' not found.")
        return {}
    except Exception as e:
        logging.error(f"Error reading CSV '{filepath}': {e}")
        return {}

def fetch_recent_news(ticker):
    """Fetches recent news (titles, summaries, links) for a ticker."""
    urls = [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
        "https://www.cnbc.com/id/19832390/device/rss/rss.html" # General CNBC Market News
    ]
    recent_articles = []
    seen_titles_or_summaries = set()
    time_threshold = datetime.now(timezone.utc) - timedelta(hours=Config.NEWS_TIME_HOURS)

    for url in urls:
        is_general_feed = 'cnbc.com' in url
        try:
            logging.debug(f"Fetching news for {ticker} from: {url}")
            response = http_session.get(url, timeout=7)
            response.raise_for_status()
            feed = feedparser.parse(response.content)
            time.sleep(Config.NEWS_FETCH_DELAY_SECONDS)

            for entry in feed.entries:
                title = getattr(entry, 'title', None)
                summary = getattr(entry, 'summary', None) or getattr(entry, 'description', None)
                link = getattr(entry, 'link', '#')
                published_parsed = getattr(entry, 'published_parsed', None)

                content_to_check = summary if summary else title
                if not content_to_check or content_to_check in seen_titles_or_summaries:
                    continue

                if is_general_feed:
                    text_for_relevance = summary if summary else title
                    if not text_for_relevance or not re.search(rf'\b{re.escape(ticker)}\b', text_for_relevance, re.IGNORECASE):
                        logging.debug(f"Skipping general article (no ticker match): {title}")
                        continue

                published_dt = None
                if published_parsed:
                    try:
                        published_dt = datetime.fromtimestamp(time.mktime(published_parsed), tz=timezone.utc)
                    except (TypeError, ValueError):
                        logging.warning(f"Could not parse date for article: {title}")
                        continue # Skip if date is unparseable

                if published_dt is None or published_dt >= time_threshold:
                    clean_title = html.unescape(title).strip() if title else "No Title"
                    clean_summary = html.unescape(re.sub('<[^<]+?>', '', summary)).strip() if summary else ""

                    recent_articles.append({
                        'title': clean_title,
                        'summary': clean_summary,
                        'link': link
                    })
                    seen_titles_or_summaries.add(content_to_check)

        except requests.exceptions.Timeout:
            logging.warning(f"Timeout fetching news from {url} for {ticker}.")
        except requests.exceptions.RequestException as e:
            status_code = e.response.status_code if e.response is not None else 'N/A'
            log_level = logging.WARNING if status_code not in [404, 403, 401] else logging.DEBUG
            logging.log(log_level, f"Error fetching news {url} for {ticker} (Status: {status_code}): {e}")
        except Exception as e:
            logging.warning(f"Error parsing news feed {url} for {ticker}: {e}")

    # Prioritize Yahoo-specific news (which comes first in urls list)
    recent_articles.sort(key=lambda x: 'feeds.finance.yahoo.com' not in x.get('link',''))
    logging.debug(f"Found {len(recent_articles)} relevant articles for {ticker} before limiting.")
    return recent_articles[:Config.MAX_NEWS_ARTICLES_PER_TICKER]

def fetch_yfinance_info(ticker):
    """Fetches the .info dictionary for a ticker from yfinance."""
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        time.sleep(Config.YFINANCE_FETCH_DELAY_SECONDS)
        if info and info.get('marketCap') is not None: # Basic check for valid data
            return info
        else:
            logging.warning(f"Incomplete or empty yfinance info dict for {ticker}.")
            return None
    except requests.exceptions.HTTPError as e:
        logging.warning(f"HTTPError fetching yfinance info for {ticker}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error fetching yfinance info for {ticker}: {e}")
        return None

# --- Cache Functions ---
def load_news_cache(filename):
    """Loads the news hash cache from a JSON file."""
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                cache = json.load(f)
                logging.info(f"Loaded news cache with {len(cache)} entries.")
                return cache
    except (IOError, json.JSONDecodeError) as e:
        logging.warning(f"Could not load news cache from {filename}: {e}")
    return {}

def save_news_cache(cache, filename):
    """Saves the news hash cache to a JSON file."""
    try:
        with open(filename, 'w') as f:
            json.dump(cache, f, indent=2)
        logging.info(f"News cache saved to {filename}.")
    except IOError as e:
        logging.error(f"Failed to save news cache to {filename}: {e}")

# --- Market Sentiment Analysis ---
def get_market_sentiment_with_gemini(model, verbose=False):
    """Fetches news for market indices (SPY, QQQ, VIX) and asks Gemini for a market sentiment analysis."""
    logging.info("--- Starting Market Sentiment Analysis ---")
    indices = {'SPY': "S&P 500", 'QQQ': "NASDAQ 100", 'VIX': "Volatility Index"}
    all_headlines = []
    for ticker, name in indices.items():
        logging.info(f"Fetching news for {ticker} ({name})...")
        articles = fetch_recent_news(ticker)
        headlines = [f"({ticker}) {a['title']}" for a in articles]
        all_headlines.extend(headlines)
        time.sleep(Config.NEWS_FETCH_DELAY_SECONDS)

    if not all_headlines:
        logging.warning("No market index news found. Skipping sentiment analysis.")
        return None

    headline_str = "\n".join([f"- {h}" for h in all_headlines])
    prompt = f"""
    Analyze the following recent market-wide news headlines (from S&P 500, NASDAQ 100, and VIX):
    {headline_str}

    Based *only* on these headlines, provide a concise (1-2 sentence) summary of the current overall market sentiment (e.g., Bullish, Bearish, Mixed, Cautious) and a brief justification.
    Also provide a simple 'recommendation' (e.g., "Risk-On", "Risk-Off", "Neutral/Selective").

    Return ONLY JSON in the format:
    {{
      "sentiment": "...",
      "justification": "...",
      "recommendation": "..."
    }}
    """

    if verbose:
        logging.debug(f"--- MARKET SENTIMENT PROMPT --- \n{prompt}\n--- END PROMPT ---")

    try:
        config = genai.types.GenerationConfig(response_mime_type="application/json")
        request_options = {"timeout": 60}
        response = model.generate_content(prompt, generation_config=config, request_options=request_options)
        
        if verbose:
            logging.debug(f"--- MARKET SENTIMENT RESPONSE --- \n{response.text}\n--- END RESPONSE ---")

        cleaned_text = response.text.strip().lstrip('```json').rstrip('```').strip()
        sentiment_data = json.loads(cleaned_text)
        
        if all(k in sentiment_data for k in ["sentiment", "justification", "recommendation"]):
            logging.info(f"Market Sentiment: {sentiment_data.get('sentiment', 'N/A')}")
            logging.info(f"Market Justification: {sentiment_data.get('justification', 'N/A')}")
            logging.info(f"Market Recommendation: {sentiment_data.get('recommendation', 'N/A')}")
            logging.info("--- Market Sentiment Analysis Complete ---")
            return sentiment_data
        else:
            logging.warning("Market sentiment analysis returned incomplete JSON.")
            return None
            
    except Exception as e:
        logging.error(f"Error during Gemini market sentiment call: {e}", exc_info=verbose)
        return None

# --- Ticker-Specific Analysis Functions ---

def format_data_for_prompt(ticker, news_articles, yf_info):
    """
    Formats news (using summaries) and yfinance data for the Gemini prompt.
    Returns the prompt string AND a dictionary of the *dynamic* yf data used.
    """
    prompt_lines = []
    yf_dynamic_data_used = {}

    def safe_get(key, default="N/A", format_spec=None, is_dynamic=False):
        """Helper to safely get, format, and track dynamic yf_info values."""
        val = yf_info.get(key)
        original_val = val
        display_val = default
        
        is_invalid = val is None or (isinstance(val, float) and math.isnan(val)) or str(val) in ['Infinity', '-Infinity']

        if not is_invalid:
            display_val = str(val)
            if format_spec:
                try:
                    # Special formatting for large numbers
                    if key in ['marketCap', 'freeCashflow', 'totalCash', 'totalDebt'] and isinstance(val, (int, float)) and val > 1_000_000:
                        if val > 1e12: display_val = f"{val/1e12:.2f}T"
                        elif val > 1e9: display_val = f"{val/1e9:.2f}B"
                        else: display_val = f"{val/1e6:.2f}M"
                    else:
                        display_val = format(val, format_spec)
                except (TypeError, ValueError):
                    display_val = str(val) # Fallback to string
        
        if is_dynamic:
            yf_dynamic_data_used[key] = original_val if not is_invalid else None
            
        return display_val
    # --- End safe_get ---

    long_name = yf_info.get('longName', ticker)
    prompt_lines.append(f"Analyze the short-term (1-day) outlook for {ticker} ({long_name}).")
    prompt_lines.append(f"Current Date: {date.today().isoformat()}\n")

    # **1. General Overview:**
    prompt_lines.append("**1. General Overview:**")
    prompt_lines.append(f"- Sector/Industry: {yf_info.get('sector', 'N/A')} / {yf_info.get('industry', 'N/A')}")
    prompt_lines.append(f"- Market Cap: {safe_get('marketCap', 'N/A', is_dynamic=True)}")
    prompt_lines.append(f"- Business: {yf_info.get('longBusinessSummary', 'N/A')[:250]}...") # Truncated
    prompt_lines.append(f"- Inst/Insider Hold: {safe_get('heldPercentInstitutions', 'N/A', '.1%', is_dynamic=True)} / {safe_get('heldPercentInsiders', 'N/A', '.1%', is_dynamic=True)}\n")

    # **2. Recent News & Volume:**
    prompt_lines.append("**2. Recent News Summaries/Headlines (Last 24h):**")
    if news_articles:
        for article in news_articles:
            # Prioritize summary, fallback to title
            content_to_send = article.get('summary') if article.get('summary') else article.get('title', 'N/A')
            prompt_lines.append(f"- {content_to_send}")
    else:
        prompt_lines.append("- No recent news found.")

    vol_str = safe_get('volume', None, is_dynamic=True)
    avg_vol_str = safe_get('averageVolume', None, is_dynamic=True)
    vol_ratio_str = "N/A"
    vol = None
    avg_vol = None
    try:
        vol = float(vol_str) if vol_str is not None and vol_str != "N/A" else None
        avg_vol = float(avg_vol_str) if avg_vol_str is not None and avg_vol_str != "N/A" else None
        if vol is not None and avg_vol is not None and avg_vol > 1e-6:
            vol_ratio = vol / avg_vol
            vol_ratio_str = f"{vol_ratio:.1f}x"
    except (ValueError, TypeError):
        pass # vol_ratio_str remains "N/A"
    prompt_lines.append(f"- Today's Volume vs Avg: {vol_ratio_str}\n")

    # **3. Performance & Growth:**
    prompt_lines.append("**3. Performance & Growth:**")
    prompt_lines.append(f"- Revenue Growth (YoY): {safe_get('revenueGrowth', 'N/A', '.1%', is_dynamic=True)}")
    prompt_lines.append(f"- Earnings Growth (YoY): {safe_get('earningsGrowth', 'N/A', '.1%', is_dynamic=True)}")
    prompt_lines.append(f"- Gross/Op Margins: {safe_get('grossMargins', 'N/A', '.1%', is_dynamic=True)} / {safe_get('operatingMargins', 'N/A', '.1%', is_dynamic=True)}")
    prompt_lines.append(f"- ROE: {safe_get('returnOnEquity', 'N/A', '.1%', is_dynamic=True)}")
    prompt_lines.append(f"- Revenue/Share: {safe_get('revenuePerShare', 'N/A', '.2f', is_dynamic=True)}")
    prompt_lines.append(f"- 52W Perf: Stock {safe_get('52WeekChange', 'N/A', '.1%', is_dynamic=True)} vs S&P {safe_get('SandP52WeekChange', 'N/A', '.1%', is_dynamic=True)}\n")

    # **4. Valuation & Outlook:**
    prompt_lines.append("**4. Valuation & Outlook:**")
    prompt_lines.append(f"- Current Price: {safe_get('currentPrice', 'N/A', '.2f', is_dynamic=True)} (Day Range: {safe_get('dayLow', 'N/A', '.2f', is_dynamic=True)} - {safe_get('dayHigh', 'N/A', '.2f', is_dynamic=True)})")
    prompt_lines.append(f"- Analyst Target Price: {safe_get('targetMeanPrice', 'N/A', '.2f', is_dynamic=True)}")
    prompt_lines.append(f"- Analyst Consensus: {safe_get('recommendationKey', 'N/A', is_dynamic=True).upper()}")
    prompt_lines.append(f"- P/E (Trailing/Fwd): {safe_get('trailingPE', 'N/A', '.2f', is_dynamic=True)} / {safe_get('forwardPE', 'N/A', '.2f', is_dynamic=True)}")
    prompt_lines.append(f"- PEG Ratio: {safe_get('pegRatio', 'N/A', '.2f', is_dynamic=True)}")
    prompt_lines.append(f"- Fwd EPS Est: {safe_get('forwardEps', 'N/A', '.2f', is_dynamic=True)}\n")

    # **5. Financial Health & Risks:**
    prompt_lines.append("**5. Financial Health & Risks:**")
    prompt_lines.append(f"- Total Cash: {safe_get('totalCash', 'N/A', is_dynamic=True)}")
    prompt_lines.append(f"- Total Debt: {safe_get('totalDebt', 'N/A', is_dynamic=True)}")
    prompt_lines.append(f"- Debt/Equity: {safe_get('debtToEquity', 'N/A', '.2f', is_dynamic=True)}")
    prompt_lines.append(f"- Free Cash Flow: {safe_get('freeCashflow', 'N/A', is_dynamic=True)}")
    
    payout_ratio_str = safe_get('payoutRatio', None, is_dynamic=True)
    payout_num = None
    try:
        if payout_ratio_str is not None and payout_ratio_str != "N/A":
            payout_num = float(payout_ratio_str)
    except (ValueError, TypeError):
        pass # Keep payout_num as None
    payout_str = f"{payout_num:.1%}" if payout_num is not None and payout_num > 0 else "N/A (No Div)"
    prompt_lines.append(f"- Dividend Payout Ratio: {payout_str} (Yield: {safe_get('dividendYield', 0, '.2%', is_dynamic=True)})")
    prompt_lines.append("")

    # **Instructions & Output Format:**
    prompt_lines.append("**Instructions:**")
    prompt_lines.append("Based *only* on the data provided (especially the recent news and volume):")
    prompt_lines.append("1.  **Score (1-10):** news_sentiment_score (1=v.bearish, 5=neutral, 10=v.bullish).")
    prompt_lines.append("2.  **Score (1-10):** growth_score (growth potential).")
    prompt_lines.append("3.  **Score (1-10):** valuation_score (1=v.expensive, 10=v.cheap).")
    prompt_lines.append("4.  **Score (1-10):** financial_health_score (1=poor, 10=excellent).")
    prompt_lines.append("5.  **Summaries (1-2 sentences):** positive_summary and negative_summary.")
    prompt_lines.append("6.  **Overall Score (1-10):** bullishness_score (1=v.bearish, 5=neutral, 10=v.bullish for *today*).")
    prompt_lines.append("7.  **Reasoning (2-3 sentences):** Justify the *bullishness_score* based on the *most important* data points (news, volume, valuation, etc.).")
    prompt_lines.append("\n**Output Format:**")
    prompt_lines.append("Return ONLY JSON: {\"news_sentiment_score\": ..., \"growth_score\": ..., \"valuation_score\": ..., \"financial_health_score\": ..., \"positive_summary\": \"...\", \"negative_summary\": \"...\", \"bullishness_score\": ..., \"reasoning\": \"...\"}")

    return "\n".join(prompt_lines), yf_dynamic_data_used


def analyze_single_ticker_with_gemini(model, ticker, news_articles, yf_info, verbose=False):
    """
    Analyzes a SINGLE ticker using news summaries and yfinance data.
    Returns the result dictionary AND the dict of dynamic yf data used.
    Prints full prompt and response if verbose is True.
    """
    if not yf_info:
        logging.warning(f"Cannot analyze {ticker} - missing yf info.")
        error_state = {**DEFAULT_ANALYSIS_ERROR_STATE, "reasoning": "Missing yfinance data"}
        return error_state, {} # Return error state and empty yf data dict

    prompt_string, yf_dynamic_data_used = format_data_for_prompt(ticker, news_articles, yf_info)

    # --- Verbose Prompt Logging ---
    if verbose:
        logging.debug(f"--- FULL PROMPT for {ticker} ---")
        print("="*20 + f" PROMPT for {ticker} " + "="*20)
        print(prompt_string)
        print("="*20 + f" END PROMPT for {ticker} " + "="*20)
        logging.debug(f"--- END FULL PROMPT for {ticker} ---")
    else:
        logging.info(f"Sending prompt for {ticker} to Gemini...")
    # --- End Prompt Logging ---

    default_error_state = {**DEFAULT_ANALYSIS_ERROR_STATE, "reasoning": "API Error"}
    raw_response_text = ""

    try:
        config = genai.types.GenerationConfig(response_mime_type="application/json")
        request_options = {"timeout": 60}
        response = model.generate_content(prompt_string, generation_config=config, request_options=request_options)
        raw_response_text = response.text

        # --- Verbose Response Logging ---
        if verbose:
            logging.debug(f"--- RAW RESPONSE for {ticker} ---")
            # Print the raw response directly to stdout as well
            print("="*20 + f" RAW RESPONSE for {ticker} " + "="*20)
            print(raw_response_text)
            print("="*20 + f" END RAW RESPONSE for {ticker} " + "="*20)
            logging.debug(f"--- END RAW RESPONSE for {ticker} ---")
        # --- End Response Logging ---

        cleaned_text = raw_response_text.strip().lstrip('```json').rstrip('```').strip()
        result = json.loads(cleaned_text)

        # --- Validation ---
        required_keys = ["news_sentiment_score", "growth_score", "valuation_score", "financial_health_score", "positive_summary", "negative_summary", "bullishness_score", "reasoning"]
        int_keys = ["news_sentiment_score", "growth_score", "valuation_score", "financial_health_score", "bullishness_score"]
        str_keys = ["positive_summary", "negative_summary", "reasoning"]
        
        if isinstance(result, dict) and all(k in result for k in required_keys):
            valid = True
            for k in int_keys:
                if not isinstance(result.get(k), int):
                    valid = False; break
                result[k] = max(1, min(10, result[k])) # Clamp score 1-10
            if valid:
                for k in str_keys:
                    if not isinstance(result.get(k), str):
                        valid = False; break
            if valid:
                return result, yf_dynamic_data_used # Success
            else:
                error_reason="API Type Error"
        else:
            error_reason="API Format Error"
        
        logging.warning(f"Gemini {ticker} {error_reason}. Using default error state.")
        return {**default_error_state, "reasoning": error_reason}, yf_dynamic_data_used

    except json.JSONDecodeError as e:
        logging.error(f"JSON decode fail {ticker}. Snippet:'{raw_response_text[:200]}...'. Err:{e}")
        error_reason = "API JSON Decode Err"
    except Exception as e:
        logging.error(f"Gemini call error {ticker}: {e}", exc_info=verbose)
        error_reason = f"API Error: {type(e).__name__}"
    
    return {**default_error_state, "reasoning": error_reason}, yf_dynamic_data_used

# --- Main Execution ---
def main(args):
    """Main execution function."""
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Verbose logging enabled.")

    gemini_model = initialize_gemini()
    initialize_database()

    # --- Market Sentiment Logic ---
    if args.market_sentiment:
        if not gemini_model:
            logging.error("Cannot run market sentiment: Gemini model not initialized.")
        else:
            sentiment_data = get_market_sentiment_with_gemini(gemini_model, args.verbose)
            if sentiment_data:
                try:
                    with open(Config.MARKET_SENTIMENT_DB, 'w') as f:
                        json.dump(sentiment_data, f, indent=2)
                    logging.info(f"Market sentiment saved to {Config.MARKET_SENTIMENT_DB}")
                except IOError as e:
                    logging.error(f"Failed to save market sentiment: {e}")
        
        # If *only* market sentiment was requested, exit now.
        if not args.confidence_csv:
            logging.info("Market sentiment analysis complete. Exiting.")
            return

    # --- Per-Ticker Analysis Logic ---
    if not args.confidence_csv:
        logging.warning("No input CSV provided for ticker analysis. Exiting.")
        return

    ticker_confidence_map = read_top_tickers_from_csv(args.confidence_csv, args.top_n)
    if not ticker_confidence_map:
        logging.error(f"No valid tickers found in '{args.confidence_csv}'. Exiting.")
        return

    tickers_to_process = list(ticker_confidence_map.keys())

    if args.gemini:
        if not gemini_model:
            logging.error("Cannot run ticker analysis: Gemini model not initialized.")
            return

        news_cache = load_news_cache(Config.NEWS_CACHE_FILENAME)
        all_db_results = []
        
        logging.info(f"--- Starting Analysis for {len(tickers_to_process)} Tickers ---")

        for i, ticker in enumerate(tickers_to_process):
            logging.info(f"Processing {ticker} ({i+1}/{len(tickers_to_process)})...")
            
            # --- Fetch Data ---
            articles = fetch_recent_news(ticker)
            headlines_for_db = [a['title'] for a in articles]
            current_headlines_hash = str(hash(frozenset(headlines_for_db))) if articles else "no_news"
            
            yf_info = fetch_yfinance_info(ticker)
            if not yf_info:
                logging.warning(f"Skipping {ticker} due to missing yfinance data.")
                # Optionally, save an error state to the DB
                error_result = {**DEFAULT_ANALYSIS_ERROR_STATE, "reasoning": "Missing yfinance data"}
                db_entry = {
                    "ticker": ticker,
                    "model_confidence_win": ticker_confidence_map.get(ticker),
                    **error_result,
                    "headlines": [], "news_hash": "error",
                    "yf_dynamic_data": {}, "yf_static_data": {}
                }
                all_db_results.append(db_entry)
                continue

            # --- Check Cache ---
            cached_hash = news_cache.get(ticker)
            if current_headlines_hash == cached_hash and not args.force_analysis:
                logging.info(f"News hash match for {ticker}. Skipping Gemini analysis.")
                continue # Skip analysis, don't update DB or cache

            # --- Analyze with Gemini ---
            logging.info(f"Analyzing {ticker} with Gemini...")
            gemini_result, yf_dynamic_data_used = analyze_single_ticker_with_gemini(
                gemini_model, ticker, articles, yf_info, args.verbose
            )

            # Log results
            score = gemini_result.get("bullishness_score", 'N/A')
            logging.info(f"📊 Gemini Result {ticker}: Overall={score}/10")
            logging.info(f"  Reasoning: {gemini_result.get('reasoning', 'N/A')}")

            # --- Prepare DB Entry ---
            yf_static_data = {k: yf_info.get(k) for k in YF_STATIC_KEYS if yf_info.get(k) is not None}
            
            db_entry = {
                "ticker": ticker,
                "model_confidence_win": ticker_confidence_map.get(ticker),
                **gemini_result,
                "headlines": headlines_for_db,
                "news_hash": current_headlines_hash,
                "yf_dynamic_data": yf_dynamic_data_used,
                "yf_static_data": yf_static_data
            }
            all_db_results.append(db_entry)

            news_cache[ticker] = current_headlines_hash # Update cache hash

            logging.info(f"Waiting {Config.API_CALL_DELAY_SECONDS} seconds...")
            time.sleep(Config.API_CALL_DELAY_SECONDS)

        # --- Save All Results ---
        save_news_cache(news_cache, Config.NEWS_CACHE_FILENAME)
        save_results_to_db(all_db_results)
        
        logging.info(f"--- Ticker Analysis Complete ---")

    else:
        logging.info("Ticker analysis skipped (no --gemini flag).")


# --- Argument Parser Setup ---
def setup_arg_parser():
    """Configures and returns the argument parser."""
    parser = argparse.ArgumentParser(
        description="v7.1 Stock Analyzer: Fetches, analyzes, and stores stock data.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    ticker_group = parser.add_argument_group('Ticker Source (Required for Gemini)')
    ticker_group.add_argument(
        '--confidence-csv',
        type=str,
        help=f"Path to the input CSV file.\nRequires columns: '{Config.TICKER_COLUMN_NAME}' and '{Config.CONFIDENCE_COLUMN_NAME}'."
    )
    ticker_group.add_argument(
        '--top-n',
        type=int,
        default=25,
        help='Number of tickers to process from the CSV, sorted by confidence (default: 25).'
    )

    analysis_group = parser.add_argument_group('Analysis Options')
    analysis_group.add_argument(
        '--market-sentiment',
        action='store_true',
        help='Run market sentiment analysis (uses Gemini).'
    )
    analysis_group.add_argument(
        '--gemini',
        action='store_true',
        help='Run per-ticker analysis using Gemini (requires --confidence-csv & API Key).'
    )
    analysis_group.add_argument(
        '--force-analysis',
        action='store_true',
        help='Force Gemini re-analysis even if news headlines have not changed.'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose DEBUG logging, including full Gemini prompts and responses.'
    )
    return parser

# --- Main Guard ---
if __name__ == "__main__":
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()

    # Validate arguments
    if not args.market_sentiment and not (args.confidence_csv and args.gemini):
        print("Error: No action specified. Use --market-sentiment or --gemini with --confidence-csv.", file=sys.stderr)
        arg_parser.print_help(sys.stderr)
        sys.exit(1)
        
    if args.gemini and not args.confidence_csv:
        print("Error: --confidence-csv is required when using the --gemini flag.", file=sys.stderr)
        arg_parser.print_help(sys.stderr)
        sys.exit(1)

    try:
        main(args)
    except Exception as e:
        logging.error(f"An unhandled exception occurred: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logging.info("Script execution finished.")
        logging.shutdown()
        sys.exit(0)