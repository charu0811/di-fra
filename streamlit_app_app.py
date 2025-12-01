import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import glob, io, zipfile

# ========================================================
# OPTIONAL LIBRARIES (fail gracefully)
# ========================================================
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


# ========================================================
# PAGE SETTINGS
# ========================================================
st.set_page_config(layout="wide", page_title="Stable OHLC Dashboard")


# ========================================================
# FILE LOADER
# ========================================================
def load_default():
    preferred = "/mnt/data/combined_multisheet_fixed_6dp_v2.xlsx"
    try:
        df = pd.read_excel(preferred, sheet_name="Combined")
        return df
    except:
        pass

    # Try other Excel
    for f in glob.glob("/mnt/data/*.xlsx"):
        try: return pd.read_excel(f, sheet_name="Combined")
        except:
            try: return pd.read_excel(f, sheet_name=0)
            except: pass

    # Try csv
    for f in glob.glob("/mnt/data/*.csv"):
        try: return pd.read_csv(f)
        except: pass

    st.error("No usable input found.")
    st.stop()


def load_uploaded(f):
    try: return pd.read_excel(f, sheet_name="Combined")
    except:
        try: return pd.read_excel(f, sheet_name=0)
        except:
            f.seek(0)
            return pd.read_csv(f)


# ========================================================
# SAFE NUMERIC PRICE EXTRACTION
# ========================================================
def safe_price(df, col):
    ser = pd.to_numeric(df[col], errors="coerce")
    if ser.dropna().size > 0:
        return ser

    # fallback
    for c in ["close", "open", "high", "low"]:
        if c in df.columns:
            cand = pd.to_numeric(df[c], errors="coerce")
            if cand.dropna().size > 0:
                st.info(f"Column '{col}' invalid. Using '{c}' instead.")
                return cand

    st.error("No numeric columns found.")
    st.stop()


# ========================================================
# SAFE ADF
# ========================================================
def adf_test(series):
    if not HAS_STATSMODELS:
        return "statsmodels missing"

    s = series.dropna()
    if len(s) < 10:
        return "Not enough data"

    if s.nunique() <= 1:
        return "Constant series — ADF not applicable"

    try:
        res = adfuller(s)
        return {
            "adf_stat": res[0],
            "p_value": res[1],
            "lags": res[2],
            "n_obs": res[3],
            "critical_values": res[4]
        }
    except Exception as e:
        return f"ADF error: {str(e)}"


# ========================================================
# ACF / PACF
# ========================================================
def plot_acf_pacf(series):
    if not HAS_STATSMODELS:
        fig = go.Figure(); fig.update_layout(title="ACF/PACF unavailable")
        return fig, fig

    s = series.dropna()

    try:
        acf_vals = acf(s, nlags=40)
        pacf_vals = pacf(s, nlags=40)
    except:
        fig = go.Figure(); fig.update_layout(title="ACF/PACF error")
        return fig, fig

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=list(range(41)), y=acf_vals))
    fig1.update_layout(title="ACF")

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=list(range(41)), y=pacf_vals))
    fig2.update_layout(title="PACF")

    return fig1, fig2


# ========================================================
# HURST
# ========================================================
def hurst_exponent(series):
    s = series.dropna().values
    if len(s) < 50: return np.nan
    lags = range(2, 20)
    tau = [np.sqrt(np.std(s[lag:] - s[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]


# ========================================================
# SAFE REGIME CLUSTERING
# ========================================================
def cluster_regimes(df, col):
    if not HAS_SKLEARN:
        return pd.Series(np.nan, index=df.index), pd.DataFrame()

    price = pd.to_numeric(df[col], errors="coerce")
    ret = price.pct_change().fillna(0)

    mean = ret.rolling(20).mean().fillna(0)
    vol = ret.rolling(20).std().fillna(0)

    X = pd.DataFrame({"mean": mean, "vol": vol}).dropna()

    # no data -> no regimes
    if len(X) < 5:
        return pd.Series(np.nan, index=df.index), X

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    gmm = GaussianMixture(n_components=2)
    labels = gmm.fit_predict(Xs)

    # FIX: ALIGN INDEX
    regimes = pd.Series(np.nan, index=df.index)
    regimes.loc[X.index] = labels

    return regimes, X


# ========================================================
# CoV REPORT
# ========================================================
def compute_cov(df):
    df = df.copy()
    df["ret"] = df["close"].pct_change()
    r = df["ret"].dropna()
    summary = {
        "mean": r.mean(),
        "std": r.std(),
        "cov": r.std() / abs(r.mean()) if r.mean() != 0 else np.nan
    }
    corr = df[["open", "high", "low", "close"]].corr()

    inst = None
    if "source_file" in df.columns:
        rows = []
        for name, g in df.groupby("source_file"):
            rr = g["close"].pct_change().dropna()
            if len(rr) > 0:
                rows.append({
                    "instrument": name,
                    "mean": rr.mean(),
                    "std": rr.std(),
                    "cov": rr.std() / abs(rr.mean()) if rr.mean() != 0 else np.nan
                })
        inst = pd.DataFrame(rows)

    return summary, corr, inst


def zip_report(summary, corr, inst):
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("summary.csv", pd.DataFrame([summary]).to_csv(index=False))
        z.writestr("correlation.csv", corr.to_csv())
        if inst is not None:
            z.writestr("per_instrument.csv", inst.to_csv(index=False))
    mem.seek(0)
    return mem


# ========================================================
# UI START
# ========================================================
st.title("Stable OHLC Dashboard")

uploaded = st.sidebar.file_uploader("Upload Excel/CSV")
df = load_uploaded(uploaded) if uploaded else load_default()

# Normalize column names
df.columns = [c.lower().replace(" ", "_") for c in df.columns]

# Date column
date_col = next((c for c in df.columns if "date" in c or "time" in c), None)
if date_col is None:
    st.error("No date/time column found.")
    st.stop()

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col]).sort_values(date_col)

# Filter sidebar
instruments = ["All"] + sorted(df["source_file"].unique()) if "source_file" in df.columns else ["All"]
inst = st.sidebar.selectbox("Instrument", instruments)

price_col = st.sidebar.selectbox("Value Column", ["open", "high", "low", "close"])

start = st.sidebar.date_input("Start", df[date_col].min())
end = st.sidebar.date_input("End", df[date_col].max())

data = df.copy()
if inst != "All":
    data = data[data["source_file"] == inst]

data = data[(data[date_col] >= pd.Timestamp(start)) & (data[date_col] <= pd.Timestamp(end))]

# SAFE price
price = safe_price(data, price_col)
data[price_col] = price

# Returns
data["ret"] = price.pct_change()
data["logret"] = np.log(price.replace(0, np.nan)).diff()

# Metrics
adf_price = adf_test(price)
adf_ret = adf_test(data["ret"])
acf_fig, pacf_fig = plot_acf_pacf(price)
hurst = hurst_exponent(price)
regimes, feats = cluster_regimes(data, price_col)


# ========================================================
# PLOT PRICE
# ========================================================
fig = go.Figure()

for label in regimes.dropna().unique():
    seg = data[regimes == label]
    fig.add_trace(go.Scatter(
        x=seg[date_col], y=seg[price_col],
        mode="lines", name=f"Regime {label}"
    ))

if regimes.dropna().size == 0:
    fig.add_trace(go.Scatter(
        x=data[date_col], y=data[price_col],
        mode="lines", name=price_col
    ))

st.plotly_chart(fig, use_container_width=True)


# ========================================================
# STATIONARITY & ACF/PACF
# ========================================================
st.subheader("ADF Results")
st.write("Price:", adf_price)
st.write("Returns:", adf_ret)

c1, c2 = st.columns(2)
c1.plotly_chart(acf_fig, use_container_width=True)
c2.plotly_chart(pacf_fig, use_container_width=True)

st.subheader("Hurst Exponent")
st.write(f"H ≈ {hurst:.4f}")


# ========================================================
# CoV REPORT
# ========================================================
st.header("CoV Report")

if st.button("Generate CoV Report"):
    summary, corr, inst_df = compute_cov(data)
    z = zip_report(summary, corr, inst_df)

    st.download_button(
        "Download CoV ZIP",
        z,
        "cov_report.zip",
        mime="application/zip"
    )

    st.subheader("Summary")
    st.table(pd.DataFrame([summary]))

    st.subheader("Correlation")
    st.dataframe(corr)

    if inst_df is not None:
        st.subheader("Per-Instrument CoV")
        st.dataframe(inst_df)
