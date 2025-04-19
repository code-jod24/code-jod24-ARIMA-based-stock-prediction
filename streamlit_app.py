import os
from datetime import datetime, timedelta
import streamlit as st
from dotenv import load_dotenv
from sentiment_analyzer import RedditSentimentAnalyzer
from stock_data import StockDataFetcher
from plotter import SentimentPlotter

# ----------------------------------------------------------------------------
# App‑level configuration
# ----------------------------------------------------------------------------
load_dotenv()
st.set_page_config(
    page_title="Reddit Sentiment & ARIMA Predictor",
    layout="centered",
    initial_sidebar_state="auto",
)

# ----------------------------------------------------------------------------
# Initialise helpers (singletons – expensive objects live once per session)
# ----------------------------------------------------------------------------
sentiment_analyzer = RedditSentimentAnalyzer()
stock_fetcher = StockDataFetcher()
plotter = SentimentPlotter()

# ----------------------------------------------------------------------------
# UI – header & user input
# ----------------------------------------------------------------------------
st.title("📈 Reddit Sentiment & ARIMA Stock Predictor")

with st.sidebar:
    st.markdown("### Configuration")
    ticker = st.text_input("Stock symbol (e.g. AAPL)", value="AAPL").upper().strip()
    period = st.selectbox("Historical window", ["1y", "6mo", "3mo"], index=0)
    analyse_btn = st.button("Analyze & Forecast", use_container_width=True)

# ----------------------------------------------------------------------------
# Main interaction – triggered when the user hits *Analyze & Forecast*
# ----------------------------------------------------------------------------
if analyse_btn and ticker:
    with st.spinner("Crunching numbers …"):
        # 1️⃣ Sentiment
        sentiment_score = sentiment_analyzer.analyze_sentiment(ticker)

        # 2️⃣ Price history
        price_response = stock_fetcher.get_stock_data(ticker)
        if not price_response["success"]:
            st.error(price_response["error"])
            st.stop()
        current_price = price_response["data"]["current_price"]
        currency = price_response["data"].get("currency", "USD")

    # ---------------------------------------------------------------------
    # Results – metrics first
    # ---------------------------------------------------------------------
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Compound sentiment", f"{sentiment_score:+.2f}")
    with col2:
        st.metric(f"{ticker} price", f"{current_price:,.2f} {currency}")

    # ---------------------------------------------------------------------
    # Placeholder: fetch richer sentiment & price series once available
    # ---------------------------------------------------------------------
    st.divider()
    st.info(
        "Correlation plots will appear here once both sentiment history "
        "and price time‑series are wired in. For now we display headline "
        "metrics only.",
        icon="ℹ️",
    )
 
# ----------------------------------------------------------------------------
# Footer – lightweight branding & help
# ----------------------------------------------------------------------------
with st.expander("ℹ️ App details"):
    st.markdown(
        "This demo combines Reddit sentiment scoring with stock‑price data "
        "retrieved from **yfinance**. ARIMA forecasting will be re‑enabled "
        "once reliable time‑series inputs are available.")
    st.caption("© 2025 code‑jod24 • Built with Streamlit 1.34")

