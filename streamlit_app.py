import os
import streamlit as st
from dotenv import load_dotenv
from sentiment_analyzer import RedditSentimentAnalyzer
from stock_data import StockDataFetcher
from plotter import plot_sentiment_vs_price 

# ----------  App setup ----------
load_dotenv()             
st.set_page_config(page_title="Reddit Sentiment & ARIMA Predictor", layout="centered")

sentiment_analyzer = RedditSentimentAnalyzer()
stock_fetcher       = StockDataFetcher()


st.title("📈 Reddit Sentiment & ARIMA Stock Predictor")

ticker = st.text_input("Enter stock symbol (e.g. AAPL)", value="AAPL").upper()
run    = st.button("Analyze & Forecast")

if run or (ticker and st.session_state.get("autostart")):
    with st.spinner("Crunching numbers…"):
        sentiment = sentiment_analyzer.analyze_sentiment(ticker)
        price_df  = stock_fetcher.get_stock_data(ticker)

    st.subheader("Latest Reddit sentiment")
    st.metric("Compound score", f"{sentiment['compound']:.2f}",
              delta=f"{sentiment['pos']*100:.1f}% positive / {sentiment['neg']*100:.1f}% negative")

    st.subheader("Price chart vs sentiment trend")
    fig = plot_sentiment_vs_price(price_df, sentiment['history'])
    st.plotly_chart(fig, use_container_width=True)

    st.success("Done! 🎉")
