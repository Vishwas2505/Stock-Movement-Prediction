import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import yfinance as yf
from lightweight_charts_v5 import lightweight_charts_v5_component
from transformers import BertTokenizer, TFBertModel
import feedparser
import plotly.graph_objects as go

# ------------------- DATA ENGINE (STABLE & MULTI-PANE) -------------------
@st.cache_data
def get_himm_data(ticker, interval="5m"):
    try:
        symbol = ticker.strip().upper()
        # Fetch 7 days for a 60-candle buffer
        df = yf.download(symbol, period="7d", interval=interval)
        if df.empty: return None
        
        # Standardize columns for latest yfinance
        df.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in df.columns]
        df = df.reset_index()
        
        # Indicators for the 3-Pane View
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain/(loss + 1e-7))))
        
        # CRITICAL: Drop NaNs to prevent JSON "Unexpected token N" error
        return df.dropna().reset_index(drop=True)
    except Exception as e:
        st.error(f"Data Error: {e}")
        return None

@st.cache_resource
def load_himm_assets():
    # HIMM architecture: Price (GRU) + Text (BERT)
    model = tf.keras.models.load_model("model.h5", 
                                      custom_objects={"TFBertModel": TFBertModel}, 
                                      compile=False)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer

# ------------------- MAIN UI -------------------
st.set_page_config(page_title="HIMM Pro Terminal", layout="wide")

# Persistent state for markers and signals
if 'markers' not in st.session_state: st.session_state.markers = []
if 'prediction' not in st.session_state: st.session_state.prediction = None

st.sidebar.title("💠 HIMM Control")
ticker_input = st.sidebar.text_input("Ticker Symbol", "BTC-USD")
interval_input = st.sidebar.selectbox("Interval", ["1m", "5m", "15m"], index=1)
sensitivity = st.sidebar.slider("Signal Sensitivity", 0.1, 1.0, 0.5)

model, tokenizer = load_himm_assets()
data = get_himm_data(ticker_input, interval_input)

if data is not None:
    st.header(f"📈 {ticker_input} Professional Multi-Pane Terminal")
    
    # --- PREPARE DATA FOR 3 PANES ---
    ts = data['Datetime'].view(np.int64) // 10**9
    ohlc = [{"time": int(t), "open": float(o), "high": float(h), "low": float(l), "close": float(c)} for t, o, h, l, c in zip(ts, data['Open'], data['High'], data['Low'], data['Close'])]
    ema_data = [{"time": int(t), "value": float(v)} for t, v in zip(ts, data['EMA_50'])]
    rsi_data = [{"time": int(t), "value": float(v)} for t, v in zip(ts, data['RSI'])]
    vol_data = [{"time": int(t), "value": float(v), "color": "#26a69a" if c >= o else "#ef5350"} for t, v, o, c in zip(ts, data['Volume'], data['Open'], data['Close'])]

    # ===================== ADVANCED 3-PANE TRADINGVIEW UI =====================
    lightweight_charts_v5_component(
        name="himm_final_terminal",
        charts=[
            {   # PANE 1: PRICE & LIVE MARKERS
                "chart": {"layout": {"background": {"color": "#0c0d14"}, "textColor": "#d1d4dc"}, "timeScale": {"timeVisible": True, "secondsVisible": True}},
                "series": [
                    {"type": "Candlestick", "data": ohlc, "options": {"upColor": "#26a69a", "downColor": "#ef5350"}, "markers": st.session_state.markers},
                    {"type": "Line", "data": ema_data, "options": {"color": "#f2ad06", "lineWidth": 1.5, "title": "EMA 50"}}
                ],
                "height": 400
            },
            {   # PANE 2: RSI
                "chart": {"layout": {"background": {"color": "#0c0d14"}, "textColor": "#d1d4dc"}},
                "series": [{"type": "Line", "data": rsi_data, "options": {"color": "#2962ff", "lineWidth": 1, "title": "RSI"}}],
                "height": 120
            },
            {   # PANE 3: VOLUME
                "chart": {"layout": {"background": {"color": "#0c0d14"}, "textColor": "#d1d4dc"}},
                "series": [{"type": "Histogram", "data": vol_data, "options": {"title": "Volume"}}],
                "height": 120
            }
        ],
        height=700
    )

    # --- HIMM DUAL-STREAM ANALYSIS ---
    st.divider()
    if st.button("RUN HIMM AI ANALYSIS 🚀", use_container_width=True):
        window_size = 60 
        if len(data) >= window_size:
            with st.spinner("Executing Hybrid Mixing Module..."):
                # 1. Price Branch: Z-Score Scaling (Forces Pattern Detection)
                raw_p = data['Close'].tail(window_size).values
                scaled_p = (raw_p - np.mean(raw_p)) / (np.std(raw_p) + 1e-7)
                
                # 2. Text Branch: BERT News Extraction
                feed = feedparser.parse(f"https://news.google.com/rss/search?q={ticker_input}+stock")
                news_txt = feed.entries[0].title if feed.entries else "Neutral."
                tokens = tokenizer(news_txt, padding='max_length', truncation=True, max_length=32, return_tensors="tf")
                
                # 3. Model Prediction
                pred = model.predict([scaled_p.reshape(1, window_size, 1), tokens['input_ids'], tokens['attention_mask']])[0][0]
                st.session_state.prediction = float(pred)
                
                # 4. SENSITIVE SIGNAL LOGIC (Triggering arrows at 50.5% / 49.5%)
                last_ts = int(data['Datetime'].iloc[-1].timestamp())
                if pred > 0.505: # High Sensitivity Buy
                    st.session_state.markers = [{"time": last_ts, "position": "belowBar", "color": "#00ff00", "shape": "arrowUp", "text": "BUY"}]
                elif pred < 0.495: # High Sensitivity Sell
                    st.session_state.markers = [{"time": last_ts, "position": "aboveBar", "color": "#ff0000", "shape": "arrowDown", "text": "SELL"}]
                else:
                    st.session_state.markers = []
                st.rerun()

    # --- AI OUTPUT GAUGE ---
    if st.session_state.prediction is not None:
        p = st.session_state.prediction
        st.subheader(f"AI Output: {'🔥 BUY' if p > 0.505 else '📉 SELL' if p < 0.495 else '⚠️ NEUTRAL'}")
        fig = go.Figure(go.Indicator(mode="gauge+number", value=p*100, title={'text': "Confidence %"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#2962ff"},
                   'steps': [{'range': [0, 49.5], 'color': "#ef5350"}, {'range': [50.5, 100], 'color': "#26a69a"}]}))
        st.plotly_chart(fig, use_container_width=True)