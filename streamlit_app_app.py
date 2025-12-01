import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import glob, io, zipfile
from datetime import datetime

# ---------------------------------------
# OPTIONAL LIBRARIES
# ---------------------------------------
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

# ------------------------------------------------------------
# DATA LOADING (robust)
# ------------------------------------------------------------
@st.cache_data
def try_load_default():
    preferred = "/mnt/data/combined_multisheet_fixed_6dp_v2.xlsx"

    # Try preferred file
    try:
        df = pd.read_excel(preferred, sheet_name="Combined", engine="openpyxl")
        return df, preferred
    except:
        pass

    # Try all Excel files
    candidates = glob.glob("/mnt/data/*.xlsx") + glob.glob("/mnt/data/*.xls")
    for c in candidates:
        try:
            df = pd.read_excel(c, sheet_name="Combined", engine="openpyxl")
            return df, c
        except:
            try:
                df = pd.read_excel(c, sheet_name=0, engine="openpyxl")
                return df, c
            except:
                pass

    # Try CSVs
    candidates = glob.glob("/mnt/data/*.csv")
    for c in candidates:
        try:
            df = pd.read_csv(c)
            return df, c
        except:
            pass

    raise FileNotFoundError("No usable Excel or CSV found in /mnt/data.")


def load_uploaded(uploaded):
    try:
        return pd.read_excel(uploaded, sheet_name="Combined")
    except:
        try:
            return pd.read_excel(uploaded, sheet_name=0)
        except:
            uploaded.seek(0)
            return pd.read_csv(uploaded)


# ------------------------------------------------------------
# SAFE NUMERIC PRICE HANDLER
# ------------------------------------------------------------
def get_price_series(df, col_choice):
    ser_num = pd.to_numeric(df[col_choice], errors="coerce")

    # If numeric data exists
    if ser_num.dropna().size > 0:
        if ser_num.isna().any():
            st.warning(f"Column '{col_choice}' has non-numeric rows. They will be ignored.")
        return ser_num

    # Column might be date or text
    try:
        parsed = pd.to_datetime(df[col_choice], errors="coerce")
        if parsed.notna().sum() > 0:
            # If close exists, fall back to close
            if "close" in df.columns:
                st.info(f"Column '{col_choice}' looks like a date. Using 'close' instead.")
                return pd.to_numeric(df["close"], errors="coerce")
            else:
                st.error(f"Column '{col_choice}' is not numeric. Please pick numeric column.")
                st.stop()
    except:
        pass

    # Try finding numeric-like fallback
    numeric_candidates = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").dropna().size > 0]

    if numeric_candidates:
        fallback = numeric_candidates[0]
        st.warning(f"Column '{col_choice}' is invalid. Falling back to '{fallback}'.")
        return pd.to_numeric(df[fallback], errors="coerce")

    st.error("No numeric-like columns found.")
    st.stop()


# ------------------------------------------------------------
# OPTIONAL METRICS
# ------------------------------------------------------------
def adf_test(series):
    if not HAS_STATSMODELS:
        return "statsmodels not installed"
    series = series.dropna()
    if len(series) < 10:
        return "Not enough data"
    result = adfuller(series)
    return {
        "adf_stat": result[0],
        "p_value": result[1],
        "used_lags": result[2],
        "n_obs": result[3],
        "critical_values": result[4]
    }


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


def hurst_exponent(series):
    import numpy as np
    import scipy.stats as stats

    ts = np.array(series.dropna())
    N = len(ts)
    if N < 50: return np.nan

    lags = np.arange(2, 20)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]


def regime_clustering(df, price_col="close", window=20):
    if not HAS_SKLEARN:
        return pd.Series(index=df.index, dtype=float), pd.DataFrame()

    ret = df[price_col].pct_change().fillna(0)
    roll_mean = ret.rolling(window).mean().fillna(0)
    roll_vol = ret.rolling(window).std().fillna(0)

    X = pd.DataFrame({"mean": roll_mean, "vol": roll_vol}).dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gmm = GaussianMixture(n_components=2)
    labels = gmm.fit_predict(X_scaled)

    regimes = pd.Series(index=X.index, data=labels)
    return regimes, X


# ------------------------------------------------------------
# Coefficient of Variation Report
# ------------------------------------------------------------
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
        groups = []
        for name, g in df.groupby("source_file"):
            r = g["close"].pct_change().dropna()
            if len(r) > 0:
                groups.append({
                    "instrument": name,
                    "mean": r.mean(),
                    "std": r.std(),
                    "cov": r.std() / abs(r.mean()) if r.mean() != 0 else np.nan
                })
        per_inst = pd.DataFrame(groups)

    return summary, corr, per_inst


def zip_report(summary, corr, per_inst):
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("summary.csv", pd.DataFrame([summary]).to_csv(index=False))
        z.writestr("correlation.csv", corr.to_csv())
        if per_inst is not None:
            z.writestr("cov_per_instrument.csv", per_inst.to_csv(index=False))
        z.writestr("README.txt", "CoV Report generated by Streamlit")

    mem.seek(0)
    return mem


# ------------------------------------------------------------
# UI START
# ------------------------------------------------------------
st.title("OHLC Research Dashboard — Updated Version")

# Load data
try:
    df, path = try_load_default()
    st.sidebar.success(f"Loaded dataset: {path}")
except:
    uploaded = st.sidebar.file_uploader("Upload Excel/CSV")
    if uploaded:
        df = load_uploaded(uploaded)
        st.sidebar.success(f"Loaded {uploaded.name}")
    else:
        st.stop()

# Normalize
df.columns = [c.lower().replace(" ", "_") for c in df.columns]

# Date column
date_col = None
for c in df.columns:
    if any(x in c for x in ["date", "time"]):
        date_col = c
        break

if date_col is None:
    st.error("No date column found.")
    st.stop()

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col])

# Sidebar options
instruments = ["All"] + sorted(df["source_file"].unique()) if "source_file" in df.columns else ["All"]
instrument = st.sidebar.selectbox("Instrument", instruments)

value_col = st.sidebar.selectbox("Price Column", ["open", "high", "low", "close"])
start = st.sidebar.date_input("Start", df[date_col].min())
end = st.sidebar.date_input("End", df[date_col].max())

# Filtering
data = df.copy()
if instrument != "All":
    data = data[data["source_file"] == instrument]

data = data[(data[date_col] >= pd.Timestamp(start)) & (data[date_col] <= pd.Timestamp(end))]

# SAFE PRICE
price = get_price_series(data, value_col)

# ------------------------------------------------------------
# ADVANCED METRICS
# ------------------------------------------------------------
data["ret"] = price.pct_change()
data["logret"] = np.log(price).diff()

adf_price = adf_test(price)
adf_ret = adf_test(data["ret"])

fig_acf, fig_pacf = plot_acf_pacf(price)
hurst = hurst_exponent(price)
regimes, feats = regime_clustering(data, price_col=value_col)

# ------------------------------------------------------------
# KPIs
# ------------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Start", str(start))
c2.metric("End", str(end))
c3.metric("Mean Price", f"{price.mean():.6f}")

# ------------------------------------------------------------
# PRICE + REGIMES
# ------------------------------------------------------------
fig = go.Figure()

if regimes.notna().sum() > 0:
    for r in regimes.unique():
        seg = data.iloc[regimes[regimes == r].index]
        fig.add_trace(go.Scatter(
            x=seg[date_col], y=seg[value_col],
            mode="lines",
            name=f"Regime {r}"
        ))
else:
    fig.add_trace(go.Scatter(x=data[date_col], y=data[value_col], mode="lines"))

st.plotly_chart(fig, use_container_width=True)

# ADF + ACF/PACF + Hurst
st.subheader("Stationarity & Autocorrelation")
st.write("ADF (Price):", adf_price)
st.write("ADF (Returns):", adf_ret)

col1, col2 = st.columns(2)
col1.plotly_chart(fig_acf, use_container_width=True)
col2.plotly_chart(fig_pacf, use_container_width=True)

st.subheader("Hurst Exponent")
st.write(f"H ≈ {hurst:.4f}")

# ------------------------------------------------------------
# REGIME FEATURES
# ------------------------------------------------------------
st.subheader("Regime Features")
if not feats.empty:
    st.dataframe(feats.join(regimes.rename("regime")).tail(200))

# ------------------------------------------------------------
# CoV REPORT
# ------------------------------------------------------------
st.header("Download CoV Report")

if st.button("Generate Report"):
    summary, corr, per_inst = compute_cov(data)
    buf = zip_report(summary, corr, per_inst)

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
