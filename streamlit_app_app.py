
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(layout="wide", page_title="OHLC Research Dashboard", initial_sidebar_state="expanded")

@st.cache_data
def load_data(path="/mnt/data/combined_multisheet_fixed_6dp_v2.xlsx"):
    # Read Combined sheet if exists otherwise first sheet
    try:
        df = pd.read_excel(path, sheet_name="Combined", engine="openpyxl")
    except Exception:
        df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    # normalize names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # choose a date column
    date_col = None
    for c in df.columns:
        if any(k in c for k in ("date","time","timestamp")):
            date_col = c
            break
    if date_col is None:
        raise ValueError("No date-like column found in the data.")
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.sort_values(by=date_col).reset_index(drop=True)
    df = df.dropna(subset=[date_col]).copy()
    df['__date_col__'] = df[date_col]
    return df, '__date_col__'

def sma(series, window):
    return series.rolling(window=window, min_periods=1).mean()

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def true_range(df, high_col, low_col, close_col):
    prev_close = df[close_col].shift(1)
    tr1 = df[high_col] - df[low_col]
    tr2 = (df[high_col] - prev_close).abs()
    tr3 = (df[low_col] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr

def atr(df, high_col, low_col, close_col, period=14):
    tr = true_range(df, high_col, low_col, close_col)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def rolling_volatility(series, window=20):
    return series.rolling(window=window).std()

# --- UI ---
st.title("OHLC Research Dashboard — Streamlit App")
st.markdown("Interactive dashboard with industry-standard indicators: SMA, EMA, RSI, MACD, ATR, Rolling volatility.")

df, date_col = load_data()

# Sidebar controls
st.sidebar.header("Controls")
instruments = sorted(df['source_file'].unique()) if 'source_file' in df.columns else ["All"]
selected_instrument = st.sidebar.selectbox("Select source / instrument", ["All"] + instruments)
resample_option = st.sidebar.selectbox("Resample frequency (for aggregation)", ["None", "1H", "4H", "1D"])
value_column = st.sidebar.selectbox("Price column to analyze", [c for c in df.columns if c in ['close','open','high','low']] , index=0)
sma_short = st.sidebar.number_input("SMA short window", min_value=1, max_value=200, value=10)
sma_long = st.sidebar.number_input("SMA long window", min_value=1, max_value=500, value=50)
ema_span = st.sidebar.number_input("EMA span", min_value=1, max_value=200, value=20)
rsi_period = st.sidebar.number_input("RSI period", min_value=2, max_value=200, value=14)
macd_fast = st.sidebar.number_input("MACD fast", min_value=1, max_value=100, value=12)
macd_slow = st.sidebar.number_input("MACD slow", min_value=2, max_value=200, value=26)
macd_signal = st.sidebar.number_input("MACD signal", min_value=1, max_value=100, value=9)
atr_period = st.sidebar.number_input("ATR period", min_value=1, max_value=200, value=14)
vol_window = st.sidebar.number_input("Rolling vol window", min_value=2, max_value=500, value=20)
start_date = st.sidebar.date_input("Start date", value=df[date_col].min().date())
end_date = st.sidebar.date_input("End date", value=df[date_col].max().date())

# filter by instrument
data = df.copy()
if selected_instrument != "All":
    data = data[data['source_file'] == selected_instrument].copy()

# set index and filter by date range
data = data.set_index(date_col)
data = data.loc[start_date:end_date].reset_index()

# Resampling if requested
if resample_option != "None":
    rule = resample_option
    agg_dict = {}
    # prefer OHLC style resampling if columns exist
    for col in ['open','high','low','close']:
        if col in data.columns:
            if col == 'open':
                agg_dict[col] = 'first'
            elif col == 'high':
                agg_dict[col] = 'max'
            elif col == 'low':
                agg_dict[col] = 'min'
            elif col == 'close':
                agg_dict[col] = 'last'
    if not agg_dict:
        # fallback: resample close via mean
        agg_dict = {value_column: 'mean'}
    data = data.set_index(date_col).resample(rule).agg(agg_dict).dropna().reset_index()

# Compute indicators
price = data[value_column].astype(float)
data[f"sma_{sma_short}"] = sma(price, sma_short)
data[f"sma_{sma_long}"] = sma(price, sma_long)
data[f"ema_{ema_span}"] = ema(price, ema_span)
data["rsi"] = rsi(price, period=rsi_period)
macd_line, macd_signal_line, macd_hist = macd(price, fast=macd_fast, slow=macd_slow, signal=macd_signal)
data["macd_line"] = macd_line
data["macd_signal"] = macd_signal_line
data["macd_hist"] = macd_hist
if all(c in data.columns for c in ['high','low','close']):
    data["atr"] = atr(data, 'high','low','close', period=atr_period)
else:
    data["atr"] = np.nan
data[f"vol_{vol_window}"] = rolling_volatility(price, window=vol_window)

# Layout: top KPIs, main chart, indicators beneath
col1, col2, col3, col4 = st.columns(4)
col1.metric("Start", data[date_col].min().strftime("%Y-%m-%d %H:%M"))
col2.metric("End", data[date_col].max().strftime("%Y-%m-%d %H:%M"))
col3.metric("Mean "+value_column, f"{price.mean():.6f}")
col4.metric("Std "+value_column, f"{price.std():.6f}")

# Price + SMA/EMA chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=data[date_col], y=data[value_column], mode='lines', name=value_column))
fig.add_trace(go.Scatter(x=data[date_col], y=data[f"sma_{sma_short}"], mode='lines', name=f"SMA{SMA_SHORT := sma_short}"))
fig.add_trace(go.Scatter(x=data[date_col], y=data[f"sma_{sma_long}"], mode='lines', name=f"SMA{SMA_LONG := sma_long}"))
fig.add_trace(go.Scatter(x=data[date_col], y=data[f"ema_{ema_span}"], mode='lines', name=f"EMA{ema_span}"))
fig.update_layout(title=f"{value_column.upper()} with Moving Averages", xaxis_title="Date", yaxis_title="Price", height=450)
st.plotly_chart(fig, use_container_width=True)

# MACD chart
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=data[date_col], y=data['macd_line'], mode='lines', name='MACD Line'))
fig2.add_trace(go.Scatter(x=data[date_col], y=data['macd_signal'], mode='lines', name='Signal Line'))
fig2.add_trace(go.Bar(x=data[date_col], y=data['macd_hist'], name='Histogram'))
fig2.update_layout(title="MACD", height=300)
st.plotly_chart(fig2, use_container_width=True)

# RSI chart
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=data[date_col], y=data['rsi'], mode='lines', name='RSI'))
fig3.add_hline(y=70, line_dash="dash", line_color="red")
fig3.add_hline(y=30, line_dash="dash", line_color="green")
fig3.update_layout(title="RSI (overbought/oversold)", height=250, yaxis=dict(range=[0,100]))
st.plotly_chart(fig3, use_container_width=True)

# ATR and Rolling Volatility
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=data[date_col], y=data['atr'], mode='lines', name='ATR'))
fig4.add_trace(go.Scatter(x=data[date_col], y=data[f'vol_{vol_window}'], mode='lines', name=f'Rolling Vol ({vol_window})'))
fig4.update_layout(title="ATR and Rolling Volatility", height=300)
st.plotly_chart(fig4, use_container_width=True)

# Statistical summaries
st.subheader("Statistical summary & research checks")
st.write("Summary statistics for selected price column:")
st.dataframe(data[[date_col, value_column, f"sma_{sma_short}", f"sma_{sma_long}", f"ema_{ema_span}", "rsi", "macd_line", "macd_signal", "atr", f'vol_{vol_window}']].describe())

# Show raw data option
with st.expander("Show raw / processed data"):
    st.dataframe(data.head(200))

st.markdown("---")
st.header("Methodology & How to interpret results")
st.markdown("""
**Indicators included & why they are industry-standard**

- **SMA / EMA (Simple / Exponential Moving Averages)**: smooth price to identify direction and filter noise. Crossovers (short above long) often indicate bullish momentum; vice versa for bearish.
- **RSI (Relative Strength Index)**: momentum oscillator (0-100). Values above 70 indicate overbought (potential pullback); below 30 indicate oversold (potential rebound).
- **MACD (Moving Average Convergence Divergence)**: shows momentum and trend change. MACD line crossing above signal line suggests bullish momentum; histogram shows the distance.
- **ATR (Average True Range)**: measures absolute volatility; useful for position sizing and understanding regime shifts.
- **Rolling volatility (std dev)**: relative volatility over a window; helps detect clustering and volatility regimes.

**Suggested research workflow**:
1. Inspect the price + moving averages to identify trend regimes (bull/bear/sideways).
2. Use ATR / rolling volatility to detect volatility regimes; combine with price to spot breakouts.
3. Use RSI & MACD for timing signals and confirmation; avoid single-signal trading—require confluence.
4. For deeper study, compute ACF/PACF, Hurst exponent, and run stationarity tests (ADF).

**Notes on reliability**:
- These indicators are descriptive; they are not predictive by themselves.
- Always validate signals with walk-forward testing and out-of-sample evaluation.
""")
