import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import numpy as np
import random
import os
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh  # type: ignore
import statsmodels.api as sm  # type: ignore
from sklearn.metrics import mean_squared_error  # type: ignore

# Initialize session state variables if they don't exist
if 'cash' not in st.session_state:
    st.session_state.cash = 100000.0
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'data' not in st.session_state:
    st.session_state.data = None
if 'auto_run' not in st.session_state:
    st.session_state.auto_run = False
if 'uploaded_symbol' not in st.session_state:
    st.session_state.uploaded_symbol = "CSV"

# --- CSV loader ---
REQUIRED_COLS = ["Date", "Open", "High", "Low", "Close", "Volume"]

def load_csv(file):
    try:
        df = pd.read_csv(file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return pd.DataFrame()

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"CSV missing required columns: {missing}")
        return pd.DataFrame()

    try:
        df["Date"] = pd.to_datetime(df["Date"])
    except Exception as e:
        st.error(f"Could not parse Date column: {e}")
        return pd.DataFrame()

    df = df.sort_values("Date").set_index("Date")
    return df

# --- Display portfolio ---
def display_portfolio():
    if st.session_state.portfolio:
        df = pd.DataFrame(st.session_state.portfolio)
        st.table(df)
    else:
        st.write("📌 No holdings yet.")

# --- Save trade history ---
def save_trade_history():
    df = pd.DataFrame(st.session_state.trade_history)
    df.to_csv("trade_history.csv", index=False)

# --- Simulate market move ---
def simulate_market_move():
    if st.session_state.data is not None and not st.session_state.data.empty:
        last_close = float(st.session_state.data['Close'].iloc[-1])
        change_pct = random.uniform(-0.02, 0.02)
        new_close = last_close * (1 + change_pct)
        last_row = st.session_state.data.iloc[-1].copy()
        new_row = last_row.copy()
        new_row['Close'] = new_close
        new_row.name = new_row.name + timedelta(days=1)
        st.session_state.data = pd.concat([st.session_state.data, pd.DataFrame([new_row])])
        st.write(f"📌 Simulated new price: ${new_close:.2f}")

# --- AI Models ---
@st.cache_resource(show_spinner=False)
def fit_sarimax(series: pd.Series):
    model = sm.tsa.statespace.SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)
    return results

def ai_suggest(df: pd.DataFrame):
    s = df["Close"].dropna().astype(float)
    if len(s) < 60:
        return "HOLD", "Not enough data for AI model (need at least ~60 closes)."

    try:
        last_price = s.iloc[-1]

        # --- Moving Averages ---
        sma = s.rolling(window=5).mean().iloc[-1]
        wma_weights = np.array([0.05, 0.1, 0.15, 0.25, 0.45])
        wma = s.rolling(window=5).apply(lambda x: np.sum(wma_weights * x)).iloc[-1]
        ema = s.ewm(span=5, adjust=False).mean().iloc[-1]

        sma_signal = "BUY" if sma > last_price else "SELL"
        wma_signal = "BUY" if wma > last_price else "SELL"
        ema_signal = "BUY" if ema > last_price else "SELL"

        # --- SARIMAX Forecast ---
        train_size = max(int(len(s) * 0.8), 40)
        train = s.iloc[:train_size]

        res = fit_sarimax(train)
        fc = res.get_forecast(steps=1)
        mean = float(fc.predicted_mean.iloc[0])
        ci = fc.conf_int().iloc[0]
        lo, hi = float(ci[0]), float(ci[1])

        sarimax_signal = "BUY" if mean > last_price else "SELL"

        # --- Combine (majority vote) ---
        signals = [sma_signal, wma_signal, ema_signal, sarimax_signal]
        final_signal = max(set(signals), key=signals.count)
        if signals.count("BUY") == signals.count("SELL"):
            final_signal = "HOLD"

        # Confidence score
        total_models = len(signals)
        votes_for_final = signals.count(final_signal)
        confidence = (votes_for_final / total_models) * 100

        # Build explanation with sorted lines
        reason = (
            f"Confidence Score: {confidence:.0f}% {final_signal}\n\n"
            f"Votes → SMA: {sma_signal}, WMA: {wma_signal}, EMA: {ema_signal}, SARIMAX: {sarimax_signal}\n\n"
            f"SARIMAX Forecast: {mean:.2f} vs Last: {last_price:.2f}\n\n"
            f"95% Confidence Interval: {lo:.2f} – {hi:.2f}"
        )

        return final_signal, reason

    except Exception as e:
        return "HOLD", f"AI model error: {e}"
# --- Main app ---
st.title("AI.ASTRA TRADING APP")

st.sidebar.markdown(
    """
    <div style="text-align: center; padding-bottom: 10px;">
        <img src="https://i.imgur.com/C3w2PbT.png" width="120">
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.header("Data Source")
uploaded_csv = st.sidebar.file_uploader(
    "Upload CSV with columns: Date, Open, High, Low, Close, Volume",
    type=["csv"]
)

if st.sidebar.button("Load Data"):
    if uploaded_csv is None:
        st.error("Please upload a CSV first.")
    else:
        data = load_csv(uploaded_csv)
        if data is None or data.empty:
            st.error("No data loaded from CSV (check columns).")
        else:
            st.session_state.data = data
            st.session_state.uploaded_symbol = os.path.splitext(uploaded_csv.name)[0]

# Chart
if st.session_state.data is not None and not st.session_state.data.empty:
    st.line_chart(st.session_state.data['Close'])
else:
    st.write("⬅ Upload & Load CSV to see chart.")

# Auto simulation
col1, col2 = st.columns(2)
with col1:
    if st.button("Start Auto Simulation"):
        st.session_state.auto_run = True
with col2:
    if st.button("Stop Auto Simulation"):
        st.session_state.auto_run = False

if st.session_state.auto_run:
    st_autorefresh(interval=2000, limit=None, key="sim-refresh")
    simulate_market_move()

# --- AI Advisor ---
with st.container():
    col1, col2, col3 = st.columns([0.02, 0.96, 0.02])
    with col2:
        st.markdown("---")
        st.markdown("### 🧠 AI.ASTRA Suggestion")
        if st.session_state.data is not None and not st.session_state.data.empty:
            signal, reason = ai_suggest(st.session_state.data)
            st.write(f"**Suggested Action:** {signal}")
            st.caption(reason)
        else:
            st.caption("Upload and load your CSV to see AI suggestions.")
        st.markdown("---")

# --- Trading form ---
st.subheader("Trade")
action = st.selectbox("Action", ["BUY", "SELL"])
qty = st.number_input("Quantity", min_value=1, value=1)

if st.button("Execute Trade"):
    if st.session_state.data is not None and not st.session_state.data.empty:
        last_price = float(st.session_state.data['Close'].iloc[-1])
        cost = last_price * qty
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if action == "BUY":
            if cost <= st.session_state.cash:
                st.session_state.cash -= cost
                found = False
                for item in st.session_state.portfolio:
                    if item['symbol'] == st.session_state.uploaded_symbol:
                        prev_qty = item['qty']
                        item['qty'] += qty
                        item['avg_price'] = ((item['avg_price'] * prev_qty) + cost) / item['qty']
                        found = True
                        break
                if not found:
                    st.session_state.portfolio.append({'symbol': st.session_state.uploaded_symbol, 'qty': qty, 'avg_price': last_price})
                st.success(f"Bought {qty} {st.session_state.uploaded_symbol} at ${last_price:.2f}")
            else:
                st.error("Not enough cash!")
        elif action == "SELL":
            found = False
            for item in st.session_state.portfolio:
                if item['symbol'] == st.session_state.uploaded_symbol and item['qty'] >= qty:
                    item['qty'] -= qty
                    st.session_state.cash += cost
                    st.success(f"Sold {qty} {st.session_state.uploaded_symbol} at ${last_price:.2f}")
                    if item['qty'] == 0:
                        st.session_state.portfolio.remove(item)
                    found = True
                    break
            if not found:
                st.error("Not enough holdings!")

        st.session_state.trade_history.append({
            'timestamp': timestamp,
            'action': action,
            'symbol': st.session_state.uploaded_symbol,
            'qty': qty,
            'price': last_price
        })
        save_trade_history()
    else:
        st.warning("Load data first.")

# --- Display cash + portfolio ---
st.metric("💵 Cash", f"${st.session_state.cash:.2f}")
st.subheader("Portfolio")
display_portfolio()

if st.session_state.trade_history:
    st.download_button(
        "Download Trade History CSV", 
        pd.DataFrame(st.session_state.trade_history).to_csv(index=False),
        "trade_history.csv",
        "text/csv"
    )
