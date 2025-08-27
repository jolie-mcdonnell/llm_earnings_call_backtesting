import streamlit as st
import pandas as pd
from scraper import scrape_ticker, combine_all_calls
from sentiment import analyze_sentiment
from strategy import backtest_sentiment_strategy
import matplotlib.pyplot as plt

st.set_page_config(page_title="Earnings Call Sentiment Trading", layout="wide")

# --- Sidebar ---
st.sidebar.header("Configuration")
tickers_input = st.sidebar.text_area(
    "Enter tickers (comma-separated):",
    placeholder="AAPL, MSFT, GOOGL"
)

date_range = st.sidebar.date_input(
    "Select start and end dates:",
    value=[pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31")]
)

run_pipeline = st.sidebar.button("Run Pipeline")

st.title("📈 Earnings Call Sentiment Trading App")

if run_pipeline:
    # Validate tickers
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    if not tickers:
        st.warning("Please enter at least one ticker.")
        st.stop()
    
    # Validate date range
    if len(date_range) != 2:
        st.warning("Please select a start and end date.")
        st.stop()
    
    start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])

    # --- Step 1: Scrape Earnings Calls ---
    st.header("Step 1: Scrape Earnings Calls")
    all_calls_list = []
    for t in tickers:
        st.write(f"Scraping {t} from {start_date.date()} to {end_date.date()}...")
        df = scrape_ticker(ticker=t, start_date=start_date, end_date=end_date)
        all_calls_list.append(df)
    all_calls = pd.concat(all_calls_list, ignore_index=True)
    st.dataframe(all_calls.head())

    # --- Step 2: Sentiment Analysis ---
    st.header("Step 2: Sentiment Analysis")
    st.write("Running sentiment model on earnings calls...")
    all_calls = analyze_sentiment(all_calls)
    st.dataframe(all_calls.head())

    # --- Step 3: Trading Strategy Backtest ---
    st.header("Step 3: Trading Strategy Backtest")
    st.write("Backtesting sentiment-based trading strategy...")
    results = backtest_sentiment_strategy(all_calls)

    for ticker, curve in results.items():
        st.subheader(ticker)
        st.line_chart(curve)
