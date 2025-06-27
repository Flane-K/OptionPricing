import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import yfinance as yf
import seaborn as sns
from scipy.stats import norm

st.set_page_config(layout="wide", page_title="Option Pricing Visualizer")
st.title("📈 Option Pricing Visualizer")

# ------------------- Black-Scholes -------------------
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S * norm.cdf(-d1)

def greeks(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S*sigma*np.sqrt(T))
    theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)
    vega = S * norm.pdf(d1) * np.sqrt(T)
    rho = K * T * np.exp(-r*T) * norm.cdf(d2)
    return delta, gamma, theta, vega, rho


# ------------------- Underlying Stock Parameters -------------------
st.sidebar.markdown("## 🔧 Configure Parameters")

with st.sidebar.expander("📈 Underlying Stock Parameters", expanded=True):
    ticker = st.text_input("Enter Stock Ticker", value="AAPL").upper()

    # Initialize defaults
    S = 100.0
    spot_price = 100.0
    currency = "$"
    vol_est = 0.2
    rf_fetch = 0.03

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        if not hist.empty:
            spot_price = hist["Close"].iloc[-1]
            currency = "₹" if ticker.endswith(".NS") else "$"
            st.success(f"Fetched Spot Price: {currency}{spot_price:.2f}")
        else:
            st.warning("No price data. Using default spot price.")
    
    except Exception:
        st.warning("Error fetching stock price. Using default.")

    S = st.number_input("Spot Price", value=float(spot_price), min_value=0.0)

    # Estimate volatility
    try:
        hist30 = stock.history(period="30d")["Close"]
        log_ret = np.log(hist30 / hist30.shift(1)).dropna()
        vol_est = np.std(log_ret) * np.sqrt(252)
        st.success(f"Estimated Volatility: {vol_est:.2f}")
    except Exception:
        st.warning("Could not estimate volatility — using 0.20")

    sigma = st.number_input("Volatility (σ)", min_value=0.01, max_value=1.0,
                            value=round(vol_est, 2), step=0.01)

    # Risk-free rate
    try:
        rf_fetch = yf.Ticker("^IRX").history(period="1d")["Close"].iloc[-1] / 100
        st.success(f"Risk-Free Rate (3-mo T-bill): {rf_fetch:.3f}")
    except Exception:
        st.warning("Could not fetch risk-free rate — using 3%")

    r = st.number_input("Risk-Free Rate (r)", min_value=0.0, max_value=0.1,
                        value=float(rf_fetch), step=0.001)

with st.sidebar.expander("⚙️ Option Parameters", expanded=True):
    K = st.number_input("Strike Price", value=float(spot_price), min_value=0.0)
    T = st.number_input("Time to Maturity (yrs)", min_value=0.01, max_value=2.0,
                        value=0.5, step=0.01)


# ------------------- Heatmap Parameters -------------------
with st.sidebar.expander("🔥 Heatmap Parameters"):
    min_spot = st.number_input("Min Spot Price", value=80.0, key="heat_min_spot")
    max_spot = st.number_input("Max Spot Price", value=120.0, key="heat_max_spot")
    min_vol = st.number_input("Min Volatility", min_value=0.01, max_value=1.0,
                              value=0.1, step=0.01, key="heat_min_vol")
    max_vol = st.number_input("Max Volatility", min_value=0.01, max_value=1.0,
                              value=0.3, step=0.01, key="heat_max_vol")

# ------------------- Cross-Section Generator -------------------
with st.sidebar.expander("🎯 Cross-Section Generator"):
    option_type = st.selectbox("Option Type", ["Call", "Put"], key="opt_type")
    varying_param = st.selectbox("Parameter to Vary",
                                 ["Spot Price", "Strike Price", "Volatility", "Time to Maturity", "Risk-Free Rate"],
                                 key="var_param")
    y_axis_value = st.selectbox("Y-Axis Value",
                                ["Price", "Delta", "Gamma", "Theta", "Vega", "Rho"],
                                key="y_axis")


# ------------------- Tabs -------------------
tab0, tab1, tab2, tab3 = st.tabs([
    "📋 Option Summary Table", 
    "📈 3D Graphs", 
    "🔥 Heatmaps", 
    "📷 Cross-Section"
])

# ------------------- Tab 0: Option Table -------------------
with tab0:
    call_price = black_scholes(S, K, T, r, sigma, "call")
    put_price = black_scholes(S, K, T, r, sigma, "put")
    cd, cg, ct, cv, cr = greeks(S, K, T, r, sigma)
    pd, pg, pt, pv, pr = greeks(S, K, T, r, sigma)

    st.markdown("## 💹 Option Prices & Greeks")
    st.markdown("""<style>
    .option-box {
        background: rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        color: white;
    }
    .option-box h2 {
        font-size: 32px;
        margin-bottom: 10px;
    }
    .option-box h3 {
        font-size: 20px;
        margin-top: 0;
    }
    </style>""", unsafe_allow_html=True)
    call_col, put_col = st.columns(2)

    with call_col:
        st.markdown(f"""
        <div class='option-box'>
            <h1 style='color:#64b5f6; text-align: center; font-size: 36px;'>📘 Call Option</h1>
            <h3 style='font-size: 26px; margin-top: 25px; margin-bottom: 18px;'>Price: {call_price:.2f}</h3>
            <p><strong>Delta:</strong> {cd:.4f}</p>
            <p><strong>Gamma:</strong> {cg:.4f}</p>
            <p><strong>Theta:</strong> {ct:.2f}</p>
            <p><strong>Vega:</strong> {cv:.2f}</p>
            <p><strong>Rho:</strong> {cr:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    with put_col:
        st.markdown(f"""
        <div class='option-box'>
            <h1 style='color:#f06292; text-align: center; font-size: 36px;'>📕 Put Option</h1>
            <h3 style='font-size: 26px; margin-top: 25px; margin-bottom: 18px;'>Price: {put_price:.2f}</h3>
            <p><strong>Delta:</strong> {1 - cd:.4f}</p>
            <p><strong>Gamma:</strong> {pg:.4f}</p>
            <p><strong>Theta:</strong> {pt:.2f}</p>
            <p><strong>Vega:</strong> {pv:.2f}</p>
            <p><strong>Rho:</strong> {-pr:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        
# ------------------- Tab 1: 3D Graphs -------------------
def plot_3d(option_type):
    spot_range = np.linspace(0.5*S, 1.5*S, 50)
    vol_range = np.linspace(0.05, 0.5, 50)
    Spot, Vol = np.meshgrid(spot_range, vol_range)
    Z = np.vectorize(black_scholes)(Spot, K, T, r, Vol, option_type)

    fig = go.Figure(data=[go.Surface(
        x=Spot, y=Vol, z=Z, colorscale='Viridis', showscale=True)])

    fig.update_layout(
        title=f"{option_type.capitalize()} Option Price Surface",
        scene=dict(
            xaxis_title="Spot Price",
            yaxis_title="Volatility",
            zaxis_title="Option Price"
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

with tab1:
    st.plotly_chart(plot_3d("call"), use_container_width=True)
    st.plotly_chart(plot_3d("put"), use_container_width=True)

# ------------------- Tab 2: Heatmaps -------------------
def plot_heatmaps():
    spot_range = np.round(np.linspace(min_spot, max_spot, 10), 2)
    vol_range = np.round(np.linspace(min_vol, max_vol, 10), 2)

    call_prices = np.zeros((len(vol_range), len(spot_range)))
    put_prices = np.zeros((len(vol_range), len(spot_range)))

    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            call_prices[i, j] = black_scholes(spot, K, T, r, vol, "call")
            put_prices[i, j] = black_scholes(spot, K, T, r, vol, "put")

    # Call Option Heatmap
    fig_call, ax_call = plt.subplots(figsize=(8, 6))
    sns.heatmap(call_prices, 
                xticklabels=[f"{s:.2f}" for s in spot_range],
                yticklabels=[f"{v:.2f}" for v in vol_range],
                annot=True, fmt=".2f", cmap="viridis", ax=ax_call)
    ax_call.set_title("Call Option Heatmap", fontsize=14, fontweight="bold")
    ax_call.set_xlabel("Spot Price")
    ax_call.set_ylabel("Volatility")

    # Put Option Heatmap
    fig_put, ax_put = plt.subplots(figsize=(8, 6))
    sns.heatmap(put_prices, 
                xticklabels=[f"{s:.2f}" for s in spot_range],
                yticklabels=[f"{v:.2f}" for v in vol_range],
                annot=True, fmt=".2f", cmap="viridis", ax=ax_put)
    ax_put.set_title("Put Option Heatmap", fontsize=14, fontweight="bold")
    ax_put.set_xlabel("Spot Price")
    ax_put.set_ylabel("Volatility")

    # Display both
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig_call)
    with col2:
        st.pyplot(fig_put)


plot_heatmaps()

# ------------------- Tab 3: Cross-Section -------------------
with tab3:
    fixed = {"S": S, "K": K, "T": T, "r": r, "sigma": sigma}
    param_map = {
        "Spot Price": "S",
        "Strike Price": "K",
        "Volatility": "sigma",
        "Time to Maturity": "T",
        "Risk-Free Rate": "r"
    }

    var_param_key = param_map[varying_param]
    x_vals = np.linspace(0.5 * fixed[var_param_key], 1.5 * fixed[var_param_key], 100)
    y_vals = []

    for val in x_vals:
        temp = fixed.copy()
        temp[var_param_key] = val
        if y_axis_value == "Price":
            y = black_scholes(temp["S"], temp["K"], temp["T"], temp["r"], temp["sigma"], option_type)
        else:
            greeks_map = {"Delta": 0, "Gamma": 1, "Theta": 2, "Vega": 3, "Rho": 4}
            y = greeks(temp["S"], temp["K"], temp["T"], temp["r"], temp["sigma"])[greeks_map[y_axis_value]]
        y_vals.append(y)

    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, label=f"{y_axis_value} vs {varying_param}")
    ax.set_xlabel(varying_param)
    ax.set_ylabel(y_axis_value)
    ax.set_title(f"{option_type.capitalize()} Option: {y_axis_value} vs {varying_param}")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
