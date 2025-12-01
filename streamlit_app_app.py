import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import glob, io, zipfile
from datetime import datetime

# -------------------------------------------------
# Optional libraries
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


st.set_page_config(layout="wide", page_title="OHLC Research Dashboard (Final)")


# -------------------------------------------------
# Robust Data Loader
# -------------------------------------------------
@st.cache_data
def try_load_default():
    preferred = "/mnt/data/combined_multisheet_fixed_6dp_v2.xlsx"

    try:
        df = pd.read_excel(preferred, sheet_name="Combined")
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


# -------------------------------------------------
# SAFE NUMERIC SERIES EXTRACTION
# -------------------------------------------------
def get_price_series(df, col):
    ser = pd.to_numeric(df[col], errors="coerce")

    if ser.dropna().size > 0:
        return ser

    # If datetime/string column
    try:
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().sum() > 0:
            if "close" in df.columns:
                st.info(f"Column '{col}' is date-like — switching to 'close'.")
                return pd.to_numeric(df["close"], errors="coerce")
            else:
                st.error("Chosen column is non-numeric. Select proper OHLC column.")
                st.stop()
    except:
        pass

    # Fallback search
    numeric_candidates = [
        c for c in df.columns
        if pd.to_numeric(df[c], errors="coerce").dropna().size > 0
    ]
    if numeric_candidates:
        fallback = numeric_candidates[0]
        st.warning(f"Fallback: using '{fallback}' instead of '{col}'.")
        return pd.to_numeric(df[fallback], errors="coerce")

    st.error("No numeric columns found.")
    st.stop()


# -------------------------------------------------
# SAFE ADF FUNCTION
# -------------------------------------------------
def adf_test(series):
    if not HAS_STATSMODELS:
        return "statsmodels not installed"

    s = series.dropna()
    if len(s) < 10:
        return "Not enough data"

    if s.nunique() <= 1:
        return "ADF not applicable: constant series"

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


# -------------------------------------------------
# ACF, PACF
# -------------------------------------------------
def plot_acf_pacf(series):
    if not HAS_STATSMODELS:
        empty = go.Figure()
        empty.update_layout(title="ACF/PACF unavailable (statsmodels missing)")
        return empty, empty

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
# HURST EXPONENT
# -------------------------------------------------
def hurst_exponent(series):
    import scipy.stats as stats

    x = np.array(series.dropna())
    if len(x) < 50:
        return np.nan

    lags = np.arange(2, 20)
    tau = [np.sqrt(np.std(x[lag:] - x[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]


# -------------------------------------------------
# REGIME CLUSTERING
# -------------------------------------------------
def regime_clustering(df, price_col="close"):
    if not HAS_SKLEARN:
        return pd.Series([np.nan] * len(df)), pd.DataFrame()

    ret = pd.to_numeric(df[price_col], errors="coerce").pct_change().fillna(0)
    rmean = ret.rolling(20).mean().fillna(0)
    rvol = ret.rolling(20).std().fillna(0)

    X = pd.DataFrame({"mean": rmean, "vol": rvol}).dropna()

    if len(X) < 10:
        return pd.Series([np.nan] * len(df)), X

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    gmm = GaussianMixture(n_components=2)
    labels = gmm.fit_predict(Xs)

    regimes = pd.Series(index=X.index, data=labels)
    return regimes, X


# -------------------------------------------------
# COV REPORT
# -------------------------------------------------
def compute_cov(df):
    df = df.copy()
    df["ret"] = df["close"].pct_change()
    r = df["ret"].dropna()

    summary = {
        "mean_return": r.mean(),
        "std_return": r.std(),
        "cov": r.std() / abs(r.mean()) if r.mean() != 0 else np.nan,
    }

    corr = df[["open", "high", "low", "close"]].corr()

    per_inst = None
    if "source_file" in df.columns:
        arr = []
        for name, g in df.groupby("source_file"):
            rr = g["close"].pct_change().dropna()
            if len(rr) > 0:
                arr.append({
                    "instrument": name,
                    "mean": rr.mean(),
                    "std": rr.std(),
                    "cov": rr.std() / abs(rr.mean()) if rr.mean() != 0 else np.nan,
                })
        per_inst = pd.DataFrame(arr)

    return summary, corr, per_inst


def zip_report(summary, corr, per_inst):
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("summary.csv", pd.DataFrame([summary]).to_csv(index=False))
        z.writestr("correlation.csv", corr.to_csv())
        if per_inst is not None:
            z.writestr("cov_per_instrument.csv", per_inst.to_csv(index=False))
        z.writestr("README.txt", "Auto-generated CoV Report")
    mem.seek(0)
    return mem


# -------------------------------------------------
# UI START
# -------------------------------------------------
st.title("OHLC Research Dashboard — Final Stable Version")

# Load data
try:
    df, path = try_load_default()
    st.sidebar.success(f"Loaded: {path}")
except:
    uploaded = st.sidebar.file_uploader("Upload Combined Excel/CSV")
    if uploaded:
        df = load_uploaded(uploaded)
        st.sidebar.success("Loaded uploaded file")
    else:
        st.stop()

# Normalize
df.columns = [c.lower().replace(" ", "_") for c in df.columns]

# find date column
date_col = next((c for c in df.columns if "date" in c or "time" in c), None)
if date_col is None:
    st.error("No date column detected.")
    st.stop()

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col])


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
instruments = ["All"] + sorted(df["source_file"].unique()) if "source_file" in df.columns else ["All"]
inst = st.sidebar.selectbox("Instrument", instruments)

price_col = st.sidebar.selectbox("Price Column", ["open", "high", "low", "close"])

start = st.sidebar.date_input("Start", df[date_col].min())
end = st.sidebar.date_input("End", df[date_col].max())

# Filtering
data = df.copy()
if inst != "All":
    data = data[data["source_file"] == inst]

data = data[(data[date_col] >= pd.Timestamp(start)) & (data[date_col] <= pd.Timestamp(end))]

# -------------------------------------
# SAFE PRICE
# -------------------------------------
price = get_price_series(data, price_col)

# CRITICAL FIX:
data[price_col] = price  # ensure numeric for clustering

data["ret"] = price.pct_change()
data["logret"] = np.log(price.replace(0, np.nan)).diff()


# -------------------------------------------------
# METRICS
# -------------------------------------------------
adf_price = adf_test(price)
adf_ret = adf_test(data["ret"])

acf_fig, pacf_fig = plot_acf_pacf(price)
hurst = hurst_exponent(price)
regimes, feats = regime_clustering(data, price_col=price_col)


# -------------------------------------------------
# KPIs
# -------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Start", str(start))
c2.metric("End", str(end))
c3.metric(f"Mean {price_col}", f"{price.mean():.6f}")


# -------------------------------------------------
# PLOT PRICE + REGIMES
# -------------------------------------------------
fig = go.Figure()

if regimes.notna().sum() > 0:
    for r in regimes.unique():
        seg = data.iloc[regimes[regimes == r].index]
        fig.add_trace(go.Scatter(
            x=seg[date_col], y=seg[price_col],
            mode="lines", name=f"Regime {r}"
        ))
else:
    fig.add_trace(go.Scatter(
        x=data[date_col], y=data[price_col],
        mode="lines", name=price_col
    ))

fig.update_layout(title="Price with Regime Segmentation", height=450)
st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------
# STATIONARITY
# -------------------------------------------------
st.subheader("ADF Stationarity Analysis")
st.write("Price:", adf_price)
st.write("Returns:", adf_ret)

colA, colB = st.columns(2)
colA.plotly_chart(acf_fig, use_container_width=True)
colB.plotly_chart(pacf_fig, use_container_width=True)

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

if st.button("Generate CoV Report"):
    summary, corr, per_inst = compute_cov(data)
    zip_buf = zip_report(summary, corr, per_inst)

    st.success("Report Ready")

    st.download_button(
        "Download CoV ZIP",
        zip_buf,
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
