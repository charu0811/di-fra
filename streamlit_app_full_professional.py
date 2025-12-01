# streamlit_app_full_with_range.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import glob, io, zipfile, os
from datetime import datetime, timedelta

# Optional libraries
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

try:
    from arch import arch_model
    HAS_ARCH = True
except Exception:
    HAS_ARCH = False

try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    HAS_STATSM = True
except Exception:
    HAS_STATSM = False

# Page config
st.set_page_config(layout="wide", page_title="Advanced Volatility & Range Dashboard")

# -----------------------
# Utility functions
# -----------------------
def load_default_combined(path="/mnt/data/combined_fixed.xlsx"):
    p = Path(path)
    if p.exists():
        try:
            df = pd.read_excel(p, engine="openpyxl")
            return df
        except Exception as e:
            st.warning(f"Could not read default combined file: {e}")
    # fallback: try CSV combine
    csvs = sorted(glob.glob("/mnt/data/*.csv"))
    frames = []
    for f in csvs:
        try:
            tmp = pd.read_csv(f)
            tmp.columns = [c.strip().lower().replace(' ', '_') for c in tmp.columns]
            date_col = next((c for c in tmp.columns if 'date' in c or 'time' in c), None)
            if date_col:
                tmp[date_col] = pd.to_datetime(tmp[date_col], errors='coerce')
            tmp['source_file'] = Path(f).stem
            frames.append(tmp)
        except Exception:
            continue
    if frames:
        comb = pd.concat(frames, ignore_index=True, sort=False)
        date_col = next((c for c in comb.columns if 'date' in c or 'time' in c), None)
        if date_col:
            comb = comb.dropna(subset=[date_col]).sort_values(by=date_col).reset_index(drop=True)
        return comb
    return None

def ensure_datetime_index(df):
    df = df.copy()
    date_col = next((c for c in df.columns if 'date' in c or 'time' in c), None)
    if date_col is None:
        st.error("No date/time column detected in the dataset.")
        st.stop()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col]).sort_values(by=date_col).reset_index(drop=True)
    df.set_index(date_col, inplace=True)
    return df

def safe_numeric(series):
    return pd.to_numeric(series, errors='coerce')

# ATR calculation (True Range & ATR)
def compute_true_range(df, high_col='high', low_col='low', close_col='close'):
    h = safe_numeric(df[high_col])
    l = safe_numeric(df[low_col])
    c = safe_numeric(df[close_col])
    prev_c = c.shift(1)
    tr1 = h - l
    tr2 = (h - prev_c).abs()
    tr3 = (l - prev_c).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr

def compute_atr(df, window=14, high_col='high', low_col='low', close_col='close'):
    tr = compute_true_range(df, high_col, low_col, close_col)
    atr = tr.rolling(window=window, min_periods=1).mean()
    return atr

# Rolling volatility (std of returns)
def rolling_volatility(series, window=20):
    return series.pct_change().rolling(window=window).std()

# Volatility heatmap: pivot time vs instrument or time-of-day vs date
def volatility_heatmap_dataframe(df, price_col='close', window=20, freq='D'):
    # Resample returns volatility at frequency freq and compute daily/hourly vol depending on freq
    series = safe_numeric(df[price_col]).pct_change()
    # create period column
    period_index = series.resample(freq).apply(lambda x: x.rolling(window).std().mean())
    # period_index is Series with DatetimeIndex
    return period_index

# Range clustering
def range_clustering(df, price_cols=('open','high','low','close'), rolling_window=20, n_clusters=3):
    # features: rolling range, rolling ATR, rolling vol
    dfc = df.copy()
    # ensure OHLC exist
    for c in price_cols:
        if c not in dfc.columns:
            dfc[c] = np.nan
    dfc['range'] = safe_numeric(dfc['high']) - safe_numeric(dfc['low'])
    dfc['atr_14'] = compute_atr(dfc, window=14)
    dfc['roll_vol_20'] = rolling_volatility(dfc['close'], window=20)
    feat = dfc[['range','atr_14','roll_vol_20']].dropna()
    if feat.empty or not HAS_SKLEARN:
        return None, feat
    scaler = StandardScaler()
    X = scaler.fit_transform(feat)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    labels = pd.Series(kmeans.labels_, index=feat.index)
    return labels, feat

# Volatility Regime Classifier (GMM/KMeans on rolling vol + returns)
def volatility_regime_classifier(df, price_col='close', vol_window=20, n_regimes=2):
    s = safe_numeric(df[price_col]).pct_change().fillna(0)
    feat = pd.DataFrame({
        'roll_vol': s.rolling(vol_window).std().fillna(0),
        'ret': s
    }).dropna()
    if feat.empty or not HAS_SKLEARN:
        return None, feat
    scaler = StandardScaler()
    X = scaler.fit_transform(feat)
    try:
        gmm = GaussianMixture(n_components=n_regimes, random_state=0).fit(X)
        labels = gmm.predict(X)
    except Exception:
        km = KMeans(n_clusters=n_regimes, random_state=0).fit(X)
        labels = km.labels_
    regimes = pd.Series(labels, index=feat.index)
    # compute transition matrix
    trans = pd.crosstab(regimes.shift(1).dropna(), regimes.loc[regimes.shift(1).dropna().index], normalize='index')
    return regimes, feat, trans

# Dynamic Range Compression Detector (low ATR percentile -> flag)
def dynamic_range_compression(df, atr_col='atr_14', window=14, percentile=10):
    atr = df[atr_col].dropna()
    if atr.empty:
        return pd.Series(index=df.index, data=False)
    # compute rolling percentile threshold
    thr = np.nanpercentile(atr.dropna(), percentile)
    # where ATR below threshold => compression
    comp = atr < thr
    # return boolean series aligned to df (fill missing with False)
    comp_full = pd.Series(False, index=df.index)
    comp_full.loc[atr.index] = comp
    return comp_full

# Trading-style regime classifier: combine vol regime + momentum (sign of rolling mean returns)
def trading_regime_classifier(df, price_col='close', vol_regimes=None, momentum_window=5):
    dfc = df.copy()
    dfc['ret'] = safe_numeric(dfc[price_col]).pct_change()
    dfc['mom'] = dfc['ret'].rolling(momentum_window).mean()
    mom_sign = dfc['mom'].apply(lambda x: 1 if x>0 else (-1 if x<0 else 0))
    # Combine: regime*10 + mom_sign to create interpretable classes
    if vol_regimes is None:
        combined = mom_sign.fillna(0)
        return combined, dfc
    combined = pd.Series(0, index=dfc.index)
    combined.loc[vol_regimes.index] = vol_regimes * 10 + mom_sign.loc[vol_regimes.index].fillna(0).astype(int)
    return combined, dfc

# -----------------------
# UI and main flow
# -----------------------
st.title("Volatility & Range Analysis Dashboard — Advanced Features")

# Load data (auto or upload)
df = load_default_combined("/mnt/data/combined_fixed.xlsx")
if df is None:
    uploaded = st.sidebar.file_uploader("Upload combined_fixed.xlsx or CSV", type=['xlsx','xls','csv'])
    if uploaded:
        try:
            if uploaded.name.lower().endswith('.csv'):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded, engine="openpyxl")
            st.sidebar.success(f"Loaded {uploaded.name}")
        except Exception as e:
            st.error(f"Failed to read upload: {e}")
            st.stop()
    else:
        st.info("Please upload combined_fixed.xlsx or place it in /mnt/data.")
        st.stop()
else:
    st.sidebar.success("Loaded combined_fixed.xlsx")

# Normalize columns and index
df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
try:
    df = ensure_datetime_index(df)
except Exception as e:
    st.error(f"Date parsing error: {e}")
    st.stop()

# detect OHLC and price column
available_price_cols = [c for c in ['open','high','low','close'] if c in df.columns]
if not available_price_cols:
    # try to find numeric column fallback
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        st.error("No OHLC or numeric columns found in dataset.")
        st.stop()
    price_col = st.sidebar.selectbox("Price column (fallback)", num_cols)
else:
    price_col = st.sidebar.selectbox("Price column", available_price_cols, index=available_price_cols.index('close') if 'close' in available_price_cols else 0)

# instrument filter
instruments = ["All"] + (sorted(df['source_file'].unique()) if 'source_file' in df.columns else [])
instr = st.sidebar.selectbox("Instrument", instruments)
if instr != "All":
    df_sel = df[df['source_file'] == instr].copy()
else:
    df_sel = df.copy()

# date filter
start = st.sidebar.date_input("Start date", df_sel.index.min().date())
end = st.sidebar.date_input("End date", df_sel.index.max().date())
df_sel = df_sel.loc[str(start):str(end)]

st.markdown(f"### Selected instrument: **{instr}**; Price column: **{price_col}**; Rows: **{len(df_sel)}**")

# compute derived series
df_calc = df_sel.copy()
# cast OHLC numeric
for c in ['open','high','low','close']:
    if c in df_calc.columns:
        df_calc[c] = safe_numeric(df_calc[c])
# compute returns
df_calc['ret'] = df_calc[price_col].pct_change()
# ATR
df_calc['tr'] = compute_true_range(df_calc, 'high','low','close') if {'high','low','close'}.issubset(df_calc.columns) else np.nan
df_calc['atr_14'] = compute_atr(df_calc, window=14, high_col='high', low_col='low', close_col='close') if {'high','low','close'}.issubset(df_calc.columns) else np.nan
# rolling vol
df_calc['roll_vol_20'] = rolling_volatility(df_calc[price_col], window=20)

# Panels layout
p1, p2 = st.columns([1,1])

with p1:
    st.subheader("Price & Volatility")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_calc.index, y=df_calc[price_col], name='price', line=dict(color='lightblue')))
    # overlay rolling volatility scaled for visual (secondary y)
    fig.add_trace(go.Scatter(x=df_calc.index, y=df_calc['roll_vol_20'], name='roll_vol_20', yaxis='y2', line=dict(color='orange')))
    fig.update_layout(height=350, legend=dict(orientation='h'), yaxis=dict(title='Price'), yaxis2=dict(title='Rolling Vol', overlaying='y', side='right'))
    st.plotly_chart(fig, use_container_width=True)

with p2:
    st.subheader("Rolling ATR & Range")
    if 'atr_14' in df_calc.columns and df_calc['atr_14'].notna().sum()>0:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_calc.index, y=df_calc['atr_14'], name='ATR(14)', line=dict(color='magenta')))
        fig2.add_trace(go.Scatter(x=df_calc.index, y=(df_calc['high']-df_calc['low']), name='Range', line=dict(color='lightgreen')))
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("ATR not available: OHLC columns missing.")

# Volatility Heatmap (resample daily or hourly)
st.subheader("Volatility Heatmap (resampled)")
heat_freq = st.selectbox("Heatmap frequency", ['D','W','H'], index=0, help="D=daily, W=weekly, H=hourly (if data intraday)")
vseries = df_calc[price_col].pct_change()
if vseries.dropna().empty:
    st.warning("No valid returns to compute heatmap.")
else:
    # compute volatility per period: std of returns inside each period
    try:
        vol_by_period = vseries.groupby(pd.Grouper(freq=heat_freq)).std().rename('vol')
        # pivot by month vs day-of-month for daily; for weekly/h hourly different layouts
        if heat_freq == 'D':
            heat_df = vol_by_period.to_frame().assign(month=vol_by_period.index.month, day=vol_by_period.index.day)
            pivot = heat_df.pivot_table(index='day', columns='month', values='vol', aggfunc='mean')
            fig_h = px.imshow(pivot, labels=dict(x="Month", y="Day", color="Vol"), aspect="auto")
            st.plotly_chart(fig_h, use_container_width=True)
        else:
            # simple time-series coloured heatmap: split into weeks
            heat_df = vol_by_period.reset_index()
            heat_df['period_index'] = range(len(heat_df))
            fig_lin = px.imshow([heat_df['vol'].fillna(0).values], labels=dict(x='Period', y='Vol'), aspect='auto')
            st.plotly_chart(fig_lin, use_container_width=True)
    except Exception as e:
        st.warning(f"Heatmap generation failed: {e}")

# Range clustering
st.subheader("Range Clustering (kmeans on range, ATR, rolling vol)")
k_clusters = st.sidebar.slider("Range clusters (k)", 2, 6, 3)
labels, feat = range_clustering(df_calc, price_cols=('open','high','low','close'), rolling_window=20, n_clusters=k_clusters)
if labels is None:
    st.warning("Range clustering unavailable (missing sklearn or insufficient data).")
else:
    df_calc['range_cluster'] = labels
    st.markdown("Cluster counts:")
    st.write(labels.value_counts())
    # show clusters over time
    figc = go.Figure()
    for lab in sorted(labels.unique()):
        seg = df_calc[df_calc['range_cluster']==lab]
        figc.add_trace(go.Scatter(x=seg.index, y=seg[price_col], mode='markers', name=f'cluster {lab}', marker=dict(size=4)))
    figc.update_layout(height=300)
    st.plotly_chart(figc, use_container_width=True)
    # show feature scatter
    if not feat.empty:
        fig_sc = px.scatter(feat.reset_index(), x='range', y='atr_14', color=labels.loc[feat.index].astype(str), title='Range vs ATR colored by cluster')
        st.plotly_chart(fig_sc, use_container_width=True)

# Volatility regime classifier
st.subheader("Volatility Regime Classifier (GMM/KMeans on rolling vol+returns)")
n_regimes = st.sidebar.slider("Regimes", 2, 4, 2)
regimes, feat_reg, trans = volatility_regime_classifier(df_calc, price_col=price_col, vol_window=20, n_regimes=n_regimes)
if regimes is None:
    st.warning("Volatility regime classification disabled (missing sklearn or insufficient data).")
else:
    df_calc['vol_regime'] = regimes
    st.write("Regime counts:")
    st.write(regimes.value_counts())
    st.write("Transition matrix (rows = from-state):")
    st.dataframe(trans)
    # plot regimes on price
    figr = go.Figure()
    for r in sorted(regimes.dropna().unique()):
        seg = df_calc[df_calc['vol_regime']==r]
        figr.add_trace(go.Scatter(x=seg.index, y=seg[price_col], mode='lines', name=f'Regime {r}'))
    figr.update_layout(height=350)
    st.plotly_chart(figr, use_container_width=True)
    # show volatility marching across regimes
    st.line_chart(df_calc[['roll_vol_20']].assign(regime=df_calc['vol_regime'].fillna(-1)))

# Dynamic Range Compression Detector
# Dynamic Range Compression Detector (robust replacement)
st.subheader("Dynamic Range Compression Detector (low ATR percentile detection)")
percentile_thr = st.sidebar.slider("ATR compression percentile threshold", 1, 50, 10)

# Ensure ATR exists
if 'atr_14' in df_calc.columns and df_calc['atr_14'].notna().any():
    # Create boolean compression series
    df_calc['compressed'] = dynamic_range_compression(df_calc, atr_col='atr_14', window=14, percentile=percentile_thr)

    # Count and show
    comp_count = int(df_calc['compressed'].sum())
    st.write(f"Compression count: {comp_count} points (ATR < {percentile_thr}th percentile)")

    # Plot price with markers for compression points
    comp_idx = df_calc.index[df_calc['compressed'].fillna(False)]
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=df_calc.index, y=df_calc[price_col], mode='lines', name='price'))
    if len(comp_idx) > 0:
        fig_comp.add_trace(go.Scatter(x=comp_idx, y=df_calc.loc[comp_idx, price_col], mode='markers',
                                      name='compressed', marker=dict(color='red', size=6)))
    fig_comp.update_layout(height=300)
    st.plotly_chart(fig_comp, use_container_width=True)

    # Group contiguous True values into intervals
    comp_series = df_calc['compressed'].fillna(False).astype(int)

    if comp_series.sum() == 0:
        st.write("No compression points detected with the current threshold.")
    else:
        # create group ids for contiguous True blocks
        # groups increments when value changes; we then keep only groups where value == 1
        grp = (comp_series != comp_series.shift(1)).cumsum()
        grouped = df_calc[['compressed']].assign(group=grp).loc[comp_series.index]
        # filter groups where 'compressed' is True
        true_groups = grouped[grouped['compressed'] == 1].groupby('group')

        comp_periods = []
        for _, g in true_groups:
            start_ts = g.index[0]
            end_ts = g.index[-1]
            comp_periods.append((start_ts, end_ts, len(g)))

        if comp_periods:
            comp_df = pd.DataFrame(comp_periods, columns=['start', 'end', 'duration_bars'])
            st.write("Compression intervals (start - end, duration in bars):")
            st.dataframe(comp_df)
        else:
            st.write("No continuous compression intervals detected.")

else:
    st.info("ATR not computed; compression detector not available.")


# Trading-style regime classifier
st.subheader("Trading-style Regime Classifier (combine vol regime & momentum)")
mom_w = st.sidebar.slider("Momentum window", 1, 21, 5)
if regimes is None:
    tclass, df_m = trading_regime_classifier(df_calc, price_col=price_col, vol_regimes=None, momentum_window=mom_w)
    st.warning("Volatility regimes not available; using momentum-only trading classifier.")
    df_calc['trading_regime'] = tclass
else:
    tclass, df_m = trading_regime_classifier(df_calc, price_col=price_col, vol_regimes=regimes, momentum_window=mom_w)
    df_calc['trading_regime'] = tclass
    st.write("Trading regime snapshot (value = regime*10 + momentum sign): e.g., 10 = regime 1 & positive momentum")
    st.dataframe(df_calc[['trading_regime']].dropna().tail(20))

# Provide downloads: CSVs and cluster labels
st.subheader("Download analysis artifacts")
buf = io.BytesIO()
with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
    z.writestr('df_calc_sample.csv', df_calc.head(1000).to_csv(index=True))
    if 'range_cluster' in df_calc.columns:
        z.writestr('range_clusters.csv', df_calc[['range_cluster']].dropna().to_csv())
    if 'vol_regime' in df_calc.columns:
        z.writestr('vol_regimes.csv', df_calc[['vol_regime']].dropna().to_csv())
    if 'trading_regime' in df_calc.columns:
        z.writestr('trading_regimes.csv', df_calc[['trading_regime']].dropna().to_csv())
buf.seek(0)
st.download_button("Download artifacts ZIP", buf, file_name="range_vol_analysis_artifacts.zip", mime="application/zip")

# Short explanation section
st.markdown("---")
st.subheader("How to interpret these outputs (quick guide)")
st.markdown("""
- **Volatility Heatmap**: shows how volatility clusters across calendar bins (daily/weekly/hourly). Bright cells = higher volatility.  
- **Rolling ATR**: average true range — a measure of how wide each bar is on average. Low ATR = compression.  
- **Range Clustering**: clusters periods by (range, ATR, rolling vol). Useful to label 'quiet', 'normal', 'explosive' periods.  
- **Volatility Regimes**: unsupervised regimes (GMM/KMeans) on rolling vol + returns; transition matrix shows how regimes persist or switch.  
- **Dynamic Range Compression Detector**: flags bars where ATR is below the chosen percentile — potential breakout setup when compression precedes expansion.  
- **Trading-style Regime Classifier**: combines volatility regime and short-term momentum to create practical signals (e.g., low-vol + positive momentum -> trend continuation).  
""")

st.info("All algorithms fall back gracefully if sklearn/arch/statsmodels are missing. Let me know if you want more refinements (EGARCH, regime duration statistics, breakout strategy backtest).")
