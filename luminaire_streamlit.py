# Project Luminaire: A Streamlit App for Analyzing SOXL with Bollinger Bands
# This app allows users to input queries in natural language to analyze the SOXL stock using Bollinger Bands and other historical analysis.
# luminaire_streamlit.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import spacy
from datetime import datetime

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Session state setup
if 'config' not in st.session_state:
    st.session_state.config = {'window_size': 20, 'multiplier': 2.0, 'timeframe': '6mo'}

if 'query_history' not in st.session_state:
    st.session_state.query_history = []

if 'last_df' not in st.session_state:
    st.session_state.last_df = pd.DataFrame()

# Styling
st.set_page_config(page_title="Project Luminaire", layout="centered", initial_sidebar_state="expanded")
st.markdown("<style>body { background-color: #0E1117; color: white; }</style>", unsafe_allow_html=True)

# Logo
st.image("luminaire_logo.png", width=100)
st.title("Project Luminaire")

# User input
user_query = st.text_input("Ask Luminaire a question (e.g., 'Show me last 1 year with window 15'):")

# NLP Parsing
def parse_query(query, current_config):
    doc = nlp(query.lower())
    updated_config = current_config.copy()

    for token in doc:
        if token.like_num:
            if 'day' in token.head.text or 'window' in token.head.text:
                updated_config['window_size'] = int(token.text)
            elif 'multiplier' in token.head.text or 'times' in token.head.text:
                updated_config['multiplier'] = float(token.text)
        if token.text in ['1mo', '3mo', '6mo', '1y', '2y', '5y']:
            updated_config['timeframe'] = token.text

    return updated_config

# Bollinger Band logic
def compute_bollinger_bands(df, window, multiplier):
    df['SMA'] = df['Close'].rolling(window=window).mean()
    df['STD'] = df['Close'].rolling(window=window).std()
    df['Upper'] = df['SMA'] + (df['STD'] * multiplier)
    df['Lower'] = df['SMA'] - (df['STD'] * multiplier)
    return df

def generate_insight(df):
    latest = df.iloc[-1]
    if latest['Close'] > latest['Upper']:
        return "Sell signal: SOXL is above the upper Bollinger Band."
    elif latest['Close'] < latest['Lower']:
        return "Buy signal: SOXL is below the lower Bollinger Band."
    else:
        return "Hold: SOXL is within the Bollinger Bands."

# Handle submission
if st.button("Analyze") and user_query:
    st.session_state.config = parse_query(user_query, st.session_state.config)
    st.session_state.query_history.append(user_query)

    df = yf.Ticker("SOXL").history(period=st.session_state.config['timeframe'])
    df = compute_bollinger_bands(df, st.session_state.config['window_size'], st.session_state.config['multiplier'])

    st.session_state.last_df = df

    insight = generate_insight(df)

    # Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='cyan')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], name='Upper Band', line=dict(dash='dot', color='gray')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], name='Lower Band', line=dict(dash='dot', color='gray')))
    fig.update_layout(
        plot_bgcolor='#0E1117',
        paper_bgcolor='#0E1117',
        font=dict(color='white'),
        title="SOXL Bollinger Band Chart"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.success(insight)
    st.caption(f"Config used: {st.session_state.config}")

# CSV Export
if not st.session_state.last_df.empty:
    csv = st.session_state.last_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, f"soxl_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")

# History Panel
with st.expander("Query History"):
    for q in reversed(st.session_state.query_history):
        st.markdown(f"- {q}")


