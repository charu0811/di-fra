# streamlit_app_full_professional.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import glob, io, zipfile, os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# --- PROFESSIONAL LIBRARIES (expected installed) ---
try:
    from arch import arch_model
    HAS_ARCH = True
except Exception:
    HAS_ARCH = False

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

try:
    import statsmodels.api as sm
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

try:
    import pmdarima as pm
    HAS_PMDARIMA = True
except Exception:
    HAS_PMDARIMA = False

from scipy.stats import entropy as shannon_entropy
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table as RLTable
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

st.set_page_config(layout="wide", page_title="Research Dashboard — Professional")

# ---------------------------
# Utilities
# ---------------------------
def load_combined(path="/mnt/data/combined_fixed.xlsx"):
    if Path(path).exists():
        df = pd.read_excel(path, engine="openpyxl")
        return df
    # fallback: try to combine CSVs
    csvs = sorted(glob.glob("/mnt/data/*.csv"))
    if csvs:
        frames = []
        for f in csvs:
            df = pd.read_csv(f)
            df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
            date_col = next((c for c in df.columns if 'date' in c or 'time' in c), None)
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df['source_file'] = Path(f).stem
            frames.append(df)
        comb = pd.concat(frames, ignore_index=True, sort=False)
        date_col = next((c for c in comb.columns if 'date' in c or 'time' in c), None)
        if date_col:
            comb = comb.dropna(subset=[date_col])
            comb = comb.sort_values(by=date_col).reset_index(drop=True)
        return comb
    raise FileNotFoundError("No combined_fixed.xlsx or CSVs found in /mnt/data")

def safe_numeric(s):
    return pd.to_numeric(s, errors='coerce')

def compute_returns(series):
    return series.pct_change().dropna()

def compute_shannon_entropy(returns, bins=50):
    # histogram-based Shannon entropy
    r = returns.dropna()
    if r.empty:
        return np.nan
    hist, edges = np.histogram(r, bins=bins, density=True)
    hist = hist[hist>0]
    return shannon_entropy(hist)

def ensure_price_col(df, col):
    if col not in df.columns:
        st.error(f"Column {col} not found.")
        st.stop()
    ser = safe_numeric(df[col])
    if ser.dropna().empty:
        st.error(f"Column {col} has no numeric values.")
        st.stop()
    return ser

# ---------------------------
# Analytics modules
# ---------------------------

def run_garch(returns, vol_model='GARCH', p=1, q=1):
    """Fit GARCH(1,1) to returns (assumes returns as decimal, not percent)."""
    if not HAS_ARCH:
        return None, "arch library not installed"
    # arch expects zero-mean series; use demeaned returns (or ret*100?)
    r = (returns - returns.mean()) * 100  # percent scale helps arch
    am = arch_model(r, vol='Garch', p=p, q=q, mean='Zero', dist='normal')
    res = am.fit(disp='off')
    # get conditional volatility (annualize if needed)
    cond_vol = res.conditional_volatility / 100.0  # back to decimal
    return cond_vol, res

def run_pca_on_returns(df, cols, n_components=3):
    if not HAS_SKLEARN:
        return None, "sklearn not installed"
    X = df[cols].pct_change().dropna()
    X = X.fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(Xs)
    return {'pca': pca, 'components': pcs, 'explained_variance': pca.explained_variance_ratio_, 'index': X.index}

def pca_volatility_clusters(df, price_col, n_clusters=3):
    """Build features using rolling volatility and cluster them"""
    if not HAS_SKLEARN:
        return None, "sklearn not installed"
    ret = df[price_col].pct_change().fillna(0)
    roll_vol_short = ret.rolling(10).std().fillna(0)
    roll_vol_long = ret.rolling(60).std().fillna(0)
    feat = pd.DataFrame({'v_short': roll_vol_short, 'v_long': roll_vol_long}).dropna()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(feat)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(Xs)
    labels = pd.Series(kmeans.labels_, index=feat.index)
    return labels, feat

def regime_transition_matrix(regimes_series):
    """Compute transition matrix for discrete regimes (NaNs ignored)."""
    r = regimes_series.dropna().astype(int)
    if r.empty:
        return None
    trans = pd.crosstab(r.shift(1).dropna(), r.loc[r.index[r.shift(1).notna()]], normalize='index')
    return trans

def var_fit_and_forecast(df, cols, steps=5):
    """Fit VAR and forecast; returns forecast dataframe."""
    if not HAS_STATSMODELS:
        return None, "statsmodels not installed"
    data = df[cols].pct_change().dropna()
    if data.shape[0] < 10:
        return None, "Not enough data for VAR"
    model = VAR(data)
    sel = model.select_order(15)
    # pick lag via AIC
    lag = sel.selected_orders.get('aic', 1)
    res = model.fit(lag)
    fc = res.forecast(data.values[-res.k_ar:], steps=steps)
    idx = pd.date_range(start=data.index[-1], periods=steps+1, freq=None)[1:]
    fc_df = pd.DataFrame(fc, index=idx, columns=data.columns)
    return fc_df, res

def auto_arima_forecast(series, steps=10):
    if not HAS_PMDARIMA:
        return None, "pmdarima not installed"
    s = series.dropna()
    if len(s) < 20:
        return None, "Not enough data for auto_arima"
    model = pm.auto_arima(s, seasonal=False, stepwise=True, suppress_warnings=True)
    fc = model.predict(n_periods=steps)
    idx = pd.RangeIndex(start=len(s), stop=len(s)+steps)
    return pd.Series(fc, index=idx), model

# ---------------------------
# PDF Report
# ---------------------------
def generate_pdf_report(summary_text, tables, out_path="/mnt/data/research_report.pdf"):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(out_path, pagesize=letter)
    story = []
    story.append(Paragraph("Research Dashboard Report", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 12))
    for name, df in tables.items():
        story.append(Paragraph(name, styles['Heading3']))
        # create simple table
        data = [df.columns.tolist()] + df.head(20).values.tolist()
        story.append(RLTable(data))
        story.append(Spacer(1, 12))
    doc.build(story)
    return out_path

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Research Dashboard — Professional (GARCH, PCA, VAR, Forecasting, Entropy, Transition Matrix)")

# Load data
# Try automatic load first
default_path = "/mnt/data/combined_fixed.xlsx"

df = None
if Path(default_path).exists():
    try:
        df = pd.read_excel(default_path, engine="openpyxl")
        st.sidebar.success("Loaded combined_fixed.xlsx from /mnt/data")
    except Exception as e:
        st.sidebar.warning(f"Could not load default file: {e}")

# If no auto-load, show uploader
if df is None:
    st.sidebar.warning("No combined_fixed.xlsx found. Please upload your dataset.")
    uploaded = st.sidebar.file_uploader("Upload combined_fixed.xlsx or any CSV/Excel", type=["xlsx","xls","csv"])
    
    if uploaded is not None:
        try:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded, engine="openpyxl")
            st.sidebar.success(f"Loaded: {uploaded.name}")
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")
            st.stop()
    else:
        st.stop()


# normalize cols
df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
date_col = next((c for c in df.columns if 'date' in c or 'time' in c), None)
if date_col is None:
    st.error("No date/time column found in the dataset.")
    st.stop()
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
df.set_index(date_col, inplace=True)

st.sidebar.header("Data & Filters")
instruments = ["All"] + (sorted(df['source_file'].unique()) if 'source_file' in df.columns else [])
instr = st.sidebar.selectbox("Instrument", instruments)
if instr != "All":
    df_plot = df[df['source_file']==instr].copy()
else:
    df_plot = df.copy()

price_choices = [c for c in df_plot.columns if c in ['open','high','low','close']]
if not price_choices:
    st.error("No open/high/low/close columns found.")
    st.stop()
price_col = st.sidebar.selectbox("Price column for analysis", price_choices, index=price_choices.index('close') if 'close' in price_choices else 0)

start_date = st.sidebar.date_input("Start date", df_plot.index.min().date())
end_date = st.sidebar.date_input("End date", df_plot.index.max().date())
df_plot = df_plot.loc[start_date:end_date]

# safe numeric price
price_ser = ensure_price_col(df_plot, price_col)
df_plot[price_col] = price_ser

# show price plot
st.subheader("Price Series")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[price_col], mode='lines', name=price_col))
fig.update_layout(height=350)
st.plotly_chart(fig, use_container_width=True)

# compute returns and basic stats
returns = df_plot[price_col].pct_change().dropna()
st.subheader("Basic Statistics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Observations", f"{len(returns)}")
col2.metric("Mean Return", f"{returns.mean():.6f}")
col3.metric("Std Return", f"{returns.std():.6f}")
col4.metric("Shannon Entropy", f"{compute_shannon_entropy(returns):.6f}")

# ADF
st.subheader("ADF Test")
adf_res = adf_test(df_plot[price_col])
st.write(adf_res)

# ACF / PACF
st.subheader("ACF / PACF")
acf_fig, pacf_fig = plot_acf_pacf(df_plot[price_col])
st.plotly_chart(acf_fig, use_container_width=True)
st.plotly_chart(pacf_fig, use_container_width=True)

# Hurst
st.metric("Hurst exponent (approx)", hurst_exponent(df_plot[price_col]))

# PCA on returns
st.subheader("PCA (Returns)")
pca_n = st.sidebar.slider("PCA components", 1, min(5, len(price_choices)), 3)
if HAS_SKLEARN:
    pca_res = run_pca_on_returns(df_plot, price_choices, n_components=pca_n)
    if isinstance(pca_res, dict):
        ev = pca_res['explained_variance']
        st.write("Explained variance ratio:", np.round(ev,4))
        figp = go.Figure(); figp.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(len(ev))], y=ev))
        st.plotly_chart(figp, use_container_width=True)
else:
    st.warning("sklearn not installed — PCA disabled")

# PCA volatility clusters
st.subheader("PCA-Vol Clustering (KMeans on rolling vol)")
clusters_k = st.sidebar.slider("PCA Vol clusters (k)", 2, 6, 3)
if HAS_SKLEARN:
    labels, feat = pca_volatility_clusters(df_plot, price_col, n_clusters=clusters_k)
    if labels is not None:
        df_plot['vol_cluster'] = labels
        st.dataframe(feat.tail(10))
        st.write("Cluster counts:")
        st.write(labels.value_counts())
else:
    st.warning("sklearn missing — cluster disabled")

# Regime clustering (GMM fallback: use KMeans if sklearn)
st.subheader("Regime Clustering (GMM/KMeans)")
if HAS_SKLEARN:
    # compute rolling mean/vol features
    ret = df_plot[price_col].pct_change().fillna(0)
    feat = pd.DataFrame({'mean': ret.rolling(20).mean().fillna(0), 'vol': ret.rolling(20).std().fillna(0)})
    feat = feat.dropna()
    if len(feat)>10:
        scaler = StandardScaler(); Xs = scaler.fit_transform(feat)
        try:
            from sklearn.mixture import GaussianMixture
            gmm = GaussianMixture(n_components=2, random_state=0).fit(Xs)
            labels = gmm.predict(Xs)
        except Exception:
            km = KMeans(n_clusters=2, random_state=0).fit(Xs)
            labels = km.labels_
        regimes = pd.Series(index=feat.index, data=labels)
        # align to df_plot index
        regimes_full = pd.Series(np.nan, index=df_plot.index)
        regimes_full.loc[regimes.index] = regimes
        df_plot['regime'] = regimes_full
        st.line_chart(df_plot[[price_col]].assign(regime=df_plot['regime'].fillna(-1)))
        st.write("Regime counts:", regimes_full.value_counts(dropna=True))
        # transition matrix
        trans = regime_transition_matrix(regimes_full)
        if trans is not None:
            st.subheader("Regime Transition Matrix")
            st.dataframe(trans)
else:
    st.warning("sklearn not available — regime clustering disabled")

# GARCH
st.subheader("GARCH Volatility Model")
if HAS_ARCH:
    if len(returns) < 50:
        st.warning("Not enough data for robust GARCH fit (<50).")
    else:
        cond_vol, garch_res = run_garch(returns)
        if cond_vol is not None:
            tmp = cond_vol.reindex(df_plot.index).fillna(method='ffill')
            st.line_chart(tmp)
            st.write(garch_res.summary().as_text())
else:
    st.warning("arch package not installed — GARCH disabled")

# VAR/VECM and forecasting
st.subheader("VAR / Forecasting")
var_cols = st.multiselect("Columns for VAR (percentage returns)", price_choices, default=[price_col])
fc_steps = st.sidebar.number_input("Forecast steps", min_value=1, max_value=60, value=10)
if HAS_STATSMODELS and len(var_cols) >= 1:
    try:
        fc_res, var_model = var_fit_and_forecast(df_plot, var_cols, steps=int(fc_steps))
        if fc_res is not None:
            st.subheader("VAR Forecast")
            st.dataframe(fc_res.head(fc_steps))
            st.line_chart(fc_res)
    except Exception as e:
        st.error(f"VAR error: {e}")
else:
    st.warning("statsmodels (VAR) not available — VAR disabled")

# auto-arima forecasting for selected series
st.subheader("Auto-ARIMA Forecast (single series)")
if HAS_PMDARIMA:
    try:
        s = df_plot[price_col].dropna().pct_change().dropna()
        if len(s) >= 30:
            arima_fc, arima_model = auto_arima_forecast(s, steps=int(fc_steps))
            if arima_fc is not None:
                st.line_chart(arima_fc)
                st.write("ARIMA model summary: (model attributes shown)")
                st.write(arima_model.summary())
        else:
            st.info("Not enough points for auto-ARIMA (>=30 required).")
    except Exception as e:
        st.error(f"Auto-ARIMA failed: {e}")
else:
    st.warning("pmdarima not installed — auto-ARIMA disabled")

# Entropy time series (rolling)
st.subheader("Entropy (rolling)")
try:
    roll_entropy = df_plot[price_col].pct_change().rolling(60).apply(lambda x: compute_shannon_entropy(pd.Series(x.dropna())), raw=False)
    st.line_chart(roll_entropy)
except Exception as e:
    st.warning(f"Entropy computation failed: {e}")

# PCA explained variance capture plot if available
if HAS_SKLEARN:
    st.subheader("PCA components (returns)")
    if isinstance(pca_res, dict):
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(len(pca_res['explained_variance']))],
                             y=pca_res['explained_variance']))
        st.plotly_chart(fig)

# Export PDF report
st.subheader("Export PDF Research Report")
if st.button("Generate PDF Report"):
    summary_text = f"Dataset: {len(df_plot)} rows. Price column: {price_col}. Date range: {df_plot.index.min()} to {df_plot.index.max()}.\n"
    tables = {
        "Basic stats": pd.DataFrame({
            'mean_return': [returns.mean()],
            'std_return': [returns.std()],
            'entropy': [compute_shannon_entropy(returns)]
        }),
    }
    pdf_path = "/mnt/data/research_report_professional.pdf"
    try:
        # Use reportlab to create a simple report
        styles = getSampleStyleSheet()
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        story = []
        story.append(Paragraph("Research Dashboard Report", styles['Title']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(summary_text.replace("\n","<br/>"), styles['Normal']))
        story.append(Spacer(1, 12))
        for name, df_tab in tables.items():
            story.append(Paragraph(name, styles['Heading3']))
            data = [df_tab.columns.tolist()] + df_tab.head(10).values.tolist()
            story.append(RLTable(data))
            story.append(Spacer(1, 12))
        doc.build(story)
        with open(pdf_path, "rb") as f:
            st.download_button("Download PDF report", data=f, file_name="research_report_professional.pdf", mime="application/pdf")
    except Exception as e:
        st.error(f"PDF generation failed: {e}")

st.markdown("---")
st.markdown("Notes: This professional build expects `arch`, `statsmodels`, `sklearn`, `pmdarima` installed. If any are missing the app will show warnings and disable specific features gracefully.")
