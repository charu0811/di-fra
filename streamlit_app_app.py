import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import glob, io, zipfile
from datetime import datetime

# -------------------------------------------------
# Optional imports (skip gracefully if unavailable)
# -------------------------------------------------
try:
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    HAS_STATSMODELS = True
except:
    HAS_STATSMODELS = False

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except:
    HAS_SKLEARN = False

st.set_page_config(layout="wide", page_title="OHLC Research Dashboard")


# -------------------------------------------------
# Robust Data Loader
# -------------------------------------------------
@st.cache_data
def try_load_default():
    preferred = "/mnt/data/combined_multisheet_fixed_6dp_v2.xlsx"

    try:
        df = pd.read_excel(preferred, sheet_name="Combined", engine="openpyxl")
        return df, preferred
    except:
        pass

    excel_files = glob.glob("/mnt/data/*.xlsx") + glob.glob("/mnt/data/*.xls")
    for f in excel_files:
        try:
            return pd.read_excel(f, sheet_name="Combined"), f
        except:
            try:
                return pd.read_excel(f, sheet_name=0), f
            except:
                pass

    csv_files = glob.glob("/mnt/data/*.csv")
    for f in csv_files:
        try:
            return pd.read_csv(f), f
        except:
            pass

    raise FileNotFoundError("No usable input file found in /mnt/data.")


def load_uploaded(uploaded):
    try:
        return pd.read_excel(uploaded, sheet_name="Combined")
    except:
        try:
            return pd.read_excel(uploaded, sheet_name=0)
        except:
            uploaded.seek(0)
            return pd.read_csv(uploaded)


# -------------------------------------------------
# Safe Numeric Conversion
# -------------------------------------------------
def get_price_series(df, col):
    ser = pd.to_numeric(df[col], errors="coerce")

    if ser.dropna().size > 0:
        if ser.isna().any():
            st.warning(f"Column '{col}' contains non-numeric rows; ignoring them.")
        return ser

    # If this column is datetime-like → not usable
    try:
        parsed = pd.to_datetime(df[col], errors="ignore")
        if parsed.notna().sum() > 0:
            if "close" in df.columns:
                st.info(f"'{col}' seems like a date/time column → using 'close'.")
                return pd.to_numeric(df["close"], errors="coerce")
            st.error("Column is not numeric. Pick a numeric price column.")
            st.stop()
    except:
        pass

    # Find fallback numeric-like column
    numeric_candidates = [
        c for c in df.columns
        if pd.to_numeric(df[c], errors="coerce").dropna().size > 0
    ]

    if numeric_candidates:
        fallback = numeric_candidates[0]
        st.warning(f"Falling back to numeric column '{fallback}'.")
        return pd.to_numeric(df[fallback], errors="coerce")

    st.error("No numeric columns found in dataset.")
    st.stop()


# -------------------------------------------------
# SAFE ADF TEST (crash-proof)
# -------------------------------------------------
def adf_test(series):
    if not HAS_STATSMODELS:
        return "statsmodels not installed"

    ser = series.dropna()
    if len(ser) < 10:
        return "Not enough data"

    if ser.nunique() <= 1:
        return "ADF not applicable: series is constant"

    try:
        result = adfuller(ser)
        return {
            "adf_stat": result[0],
            "p_value": result[1],
            "used_lags": result[2],
            "n_obs": result[3],
            "critical_values": result[4]
        }
    except Exception as e:
        return f"ADF error: {str(e)}"


# -------------------------------------------------
# ACF & PACF
# -------------------------------------------------
def plot_acf_pacf(series):
    if not HAS_STATSMODELS:
        fig1 = go.Figure(); fig1.update_layout(title="ACF (statsmodels missing)")
        fig2 = go.Figure(); fig2.update_layout(title="PACF (statsmodels missing)")
        return fig1, fig2

    s = series.dropna()
    acf_vals = acf(s, nlags=40)
    pacf_vals = pacf(s, nlags=40)

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals))
    fig1.update_layout(title="ACF")

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals))
    fig2.update_layout(title="PACF")

    return fig1, fig2


# -------------------------------------------------
# Hurst Exponent
# -------------------------------------------------
def hurst_exponent(series):
    import numpy as np
    import scipy.stats as stats

    x = np.array(series.dropna())
    N = len(x)
    if N < 50: return np.nan

    lags = np.arange(2, 20)
    tau = [np.sqrt(np.std(x[lag:] - x[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]


# -------------------------------------------------
# Regime Clustering
# -------------------------------------------------
def regime_clustering(df, price_col="close", window=20):
    if not HAS_SKLEARN:
        return pd.Series([np.nan] * len(df)), pd.DataFrame()

    ret = df[price_col].pct_change().fillna(0)
    rmean = ret.rolling(window).mean().fillna(0)
    rvol = ret.rolling(window).std().fillna(0)

    X = pd.DataFrame({"mean": rmean, "vol": rvol}).dropna()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    gmm = GaussianMixture(n_components=2)
    labels = gmm.fit_predict(Xs)

    regimes = pd.Series(index=X.index, data=labels)
    return regimes, X


# -------------------------------------------------
# Coefficient of Variation Report
# -------------------------------------------------
def compute_cov(df):
    df = df.copy()
    df["ret"] = df["close"].pct_change()
    ret = df["ret"].dropna()

    summary = {
        "mean_return": ret.mean(),
        "std_return": ret.std(),
        "cov": ret.std() / abs(ret.mean()) if ret.mean() != 0 else np.nan
    }

    corr = df[["open", "high", "low", "close"]].corr()

    per_inst = None
    if "source_file" in df.columns:
        rows = []
        for name, g in df.groupby("source_file"):
            r = g["close"].pct_change().dropna()
            if len(r) > 0:
                rows.append({
                    "instrument": name,
                    "mean": r.mean(),
                    "std": r.std(),
                    "cov": r.std() / abs(r.mean()) if r.mean() != 0 else np.nan
                })
        per_inst = pd.DataFrame(rows)

    return summary, corr, per_inst


def zip_report(summary, corr, per_inst):
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("summary.csv", pd.DataFrame([summary]).to_csv(index=False))
        z.writestr("correlation.csv", corr.to_csv())
        if per_inst is not None:
            z.writestr("cov_per_instrument.csv", per_inst.to_csv(index=False))
        z.writestr("README.txt", "CoV report generated automatically.")
    mem.seek(0)
    return mem


# -------------------------------------------------
# UI START
# -------------------------------------------------
st.title("OHLC Research Dashboard — Updated & Stable")

# Load data
try:
    df, path = try_load_default()
    st.sidebar.success(f"Loaded: {path}")
except:
    uploaded = st.sidebar.file_uploader("Upload Excel/CSV")
    if uploaded:
        df = load_uploaded(uploaded)
        st.sidebar.success(f"Loaded {uploaded.name}")
    else:
        st.stop()

# Normalize
df.columns = [c.lower().replace(" ", "_") for c in df.columns]

# Find date column
date_col = next((c for c in df.columns if "date" in c or "time" in c), None)
if date_col is None:
    st.error("No date column found.")
    st.stop()

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col]).sort_values(by=date_col)

# -------------------------------
# Sidebar Filters
# -------------------------------
instruments = ["All"] + sorted(df["source_file"].unique()) if "source_file" in df.columns else ["All"]
selected_inst = st.sidebar.selectbox("Instrument", instruments)

value_col = st.sidebar.selectbox("Price Column", ["open", "high", "low", "close"])

start = st.sidebar.date_input("Start Date", df[date_col].min())
end = st.sidebar.date_input("End Date", df[date_col].max())

# Apply filters
data = df.copy()
if selected_inst != "All":
    data = data[data["source_file"] == selected_inst]

data = data[(data[date_col] >= pd.Timestamp(start)) & (data[date_col] <= pd.Timestamp(end))]

# SAFE PRICE
price = get_price_series(data, value_col)
data["ret"] = price.pct_change()
data["logret"] = np.log(price).diff()

# -------------------------------------------------
# METRICS
# -------------------------------------------------
adf_price = adf_test(price)
adf_returns = adf_test(data["ret"])
acf_fig, pacf_fig = plot_acf_pacf(price)
hurst = hurst_exponent(price)
regimes, feats = regime_clustering(data, price_col=value_col)

# -------------------------------------------------
# KPIs
# -------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Start", str(start))
c2.metric("End", str(end))
c3.metric("Mean Price", f"{price.mean():.6f}")

# -------------------------------------------------
# PRICE + REGIME PLOT
# -------------------------------------------------
fig = go.Figure()

if regimes.notna().sum() > 0:
    for r in regimes.unique():
        seg = data.iloc[regimes[regimes == r].index]
        fig.add_trace(go.Scatter(
            x=seg[date_col], y=seg[value_col],
            mode="lines", name=f"Regime {r}"
        ))
else:
    fig.add_trace(go.Scatter(
        x=data[date_col], y=data[value_col],
        mode="lines", name=value_col
    ))

fig.update_layout(title="Regime Clustering", height=450)
st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# STATIONARITY
# -------------------------------------------------
st.subheader("Stationarity Tests (ADF)")

st.write("Price ADF:", adf_price)
st.write("Returns ADF:", adf_returns)

col1, col2 = st.columns(2)
col1.plotly_chart(acf_fig, use_container_width=True)
col2.plotly_chart(pacf_fig, use_container_width=True)

# -------------------------------------------------
# HURST
# -------------------------------------------------
st.subheader("Hurst Exponent")
st.write(f"H ≈ {hurst:.4f}")

# -------------------------------------------------
# REGIME FEATURES
# -------------------------------------------------
st.subheader("Regime Features")
if not feats.empty:
    st.dataframe(feats.join(regimes.rename("regime")).tail(200))

# -------------------------------------------------
# CoV REPORT
# -------------------------------------------------
st.header("Coefficient of Variation (CoV) Report")

if st.button("Generate CoV Report ZIP"):
    summary, corr, per_inst = compute_cov(data)
    buf = zip_report(summary, corr, per_inst)

    st.success("Report Generated")

    st.download_button(
        "Download CoV ZIP",
        buf,
        "cov_report.zip",
        mime="application/zip"
    )

    st.subheader("Summary")
    st.table(pd.DataFrame([summary]))

    st.subheader("Correlation Matrix")
    st.dataframe(corr)

    if per_inst is not None:
        st.subheader("CoV Per Instrument")
        st.dataframe(per_inst)
