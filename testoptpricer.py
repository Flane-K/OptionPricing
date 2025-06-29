import streamlit as st
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from scipy.stats import norm
import pandas as pd

st.set_page_config(layout="wide", page_title="Option Pricing Visualizer")

# --- Modern Glassmorphic CSS with Corrected Container Logic ---
st.markdown("""
<style>
    /* Global dark theme */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        color: #e0e0e0;
    }
    
    /* CORRECTED: Target the tab panel as the main hover container */
    [data-baseweb="tab-panel"] {
        background: transparent;
        border: 1px solid transparent;
        border-radius: 20px;
        padding: 25px;
        margin-top: 10px; /* Minimized margin */
        transition: background 0.4s ease, border-color 0.4s ease, box-shadow 0.4s ease, backdrop-filter 0.4s ease;
    }

    /* Apply the glass/hover effect to the container */
    [data-baseweb="tab-panel"]:hover {
        background: rgba(26, 26, 46, 0.7); /* More visible background on hover */
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(177, 156, 217, 0.5); /* Purple border on hover */
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    }
    
    /* Header styling changed to Purple */
    h1, h2, h3, h4, h5, h6 {
        color: #B19CD9 !important;
        text-shadow: 0 0 10px rgba(177, 156, 217, 0.3);
        font-weight: 600;
    }
    
    /* Main title styling */
    .main-title {
        background: linear-gradient(45deg, #40E0D0, #8A2BE2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem !important;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: none;
    }
    
    /* Metric value styling (Neon Bluish) */
    [data-testid="stMetricValue"] {
        color: #40E0D0 !important;
        font-size: 2rem !important;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(64, 224, 208, 0.5);
    }
    
    [data-testid="stMetricLabel"] {
        color: #B19CD9 !important;
        font-weight: 500;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(26, 26, 46, 0.8) !important;
        backdrop-filter: blur(20px);
    }
    
    /* Tab bar styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 5px;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        color: #B19CD9;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(64, 224, 208, 0.1);
        color: #40E0D0;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, rgba(64, 224, 208, 0.2), rgba(138, 43, 226, 0.2)) !important;
        color: #40E0D0 !important;
    }
    
    /* Input styling */
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(64, 224, 208, 0.3) !important;
        border-radius: 10px !important;
        color: #e0e0e0 !important;
        backdrop-filter: blur(10px);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #40E0D0, #8A2BE2) !important;
        border: none !important;
        border-radius: 15px !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.5rem 2rem !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(64, 224, 208, 0.4) !important;
    }

</style>
""", unsafe_allow_html=True)

# Title with modern styling
st.markdown('<h1 class="main-title">📈 Option Pricing Visualizer</h1>', unsafe_allow_html=True)

# ------------------- Black-Scholes Model -------------------
def black_scholes(S, K, T, r, sigma, option_type="call"):
    """Calculates the Black-Scholes option price."""
    if T <= 0 or sigma <= 0: return max(0, S - K) if option_type == 'call' else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type.lower() == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # Put
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_greeks(S, K, T, r, sigma, option_type="call"):
    """Calculates the Greeks for the Black-Scholes model."""
    if T <= 0 or sigma <= 0: return 0, 0, 0, 0, 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * pdf_d1 * np.sqrt(T) / 100 # per 1% change
    
    if option_type.lower() == "call":
        delta = norm.cdf(d1)
        theta = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365 # per day
        rho = (K * T * np.exp(-r * T) * norm.cdf(d2)) / 100 # per 1% change
    else:  # Put
        delta = norm.cdf(d1) - 1
        theta = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365 # per day
        rho = (-K * T * np.exp(-r * T) * norm.cdf(-d2)) / 100 # per 1% change
        
    return delta, gamma, theta, vega, rho

# ------------------- Binomial Option Pricing Model -------------------
def binomial_option_pricing(S, K, T, r, sigma, option_type="call", N=100):
    """Calculates option price using the Cox-Ross-Rubinstein binomial model."""
    if T <= 0: return max(0, S - K) if option_type == 'call' else max(0, K - S)
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    if p < 0 or p > 1: return np.nan # Avoid arbitrage issues

    prices = S * (d**np.arange(N, -1, -1)) * (u**np.arange(0, N + 1, 1))
    
    option_values = np.maximum(0, prices - K) if option_type.lower() == "call" else np.maximum(0, K - prices)

    for i in range(N - 1, -1, -1):
        option_values = (p * option_values[1:] + (1 - p) * option_values[:-1]) * np.exp(-r * dt)
    
    return option_values[0]

# ------------------- Monte Carlo Simulation Model -------------------
def monte_carlo_option_pricing(S, K, T, r, sigma, option_type="call", num_simulations=10000):
    """Calculates option price using Monte Carlo simulation."""
    if T <= 0: return max(0, S - K) if option_type == 'call' else max(0, K - S)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * np.random.standard_normal(num_simulations))
    payoffs = np.maximum(0, ST - K) if option_type.lower() == "call" else np.maximum(0, K - ST)
    return np.exp(-r * T) * np.mean(payoffs)

# --- Generic Greeks Calculator (Finite Differences) ---
def finite_difference_greeks(pricing_func, S, K, T, r, sigma, option_type, **kwargs):
    """Calculates Greeks using finite differences for any pricing model."""
    dS = S * 0.01; dT = 1/365; d_sigma = 0.01; d_r = 0.01
    
    p_mid = pricing_func(S, K, T, r, sigma, option_type, **kwargs)
    if T - dT <= 0:
        theta = -p_mid / dT # Approximation for last day
    else:
        p_t = pricing_func(S, K, T - dT, r, sigma, option_type, **kwargs)
        theta = p_t - p_mid
        
    p_S_up = pricing_func(S + dS, K, T, r, sigma, option_type, **kwargs)
    p_S_down = pricing_func(S - dS, K, T, r, sigma, option_type, **kwargs)
    delta = (p_S_up - p_S_down) / (2 * dS)
    gamma = (p_S_up - 2 * p_mid + p_S_down) / (dS ** 2)
    
    p_vol_up = pricing_func(S, K, T, r, sigma + d_sigma, option_type, **kwargs)
    vega = (p_vol_up - p_mid) / d_sigma
    
    p_r_up = pricing_func(S, K, T, r + d_r, sigma, option_type, **kwargs)
    rho = (p_r_up - p_mid) / d_r
    
    return delta, gamma, theta, vega, rho

# ------------------- Sidebar Controls -------------------
st.sidebar.markdown("## 🔧 Configure Parameters")
selected_model = st.sidebar.selectbox("Select Pricing Model", ["Black-Scholes", "Binomial Option Pricing", "Monte Carlo Simulation"])

model_params = {}
pricing_function = black_scholes
if selected_model == "Binomial Option Pricing":
    model_params['N'] = st.sidebar.slider("Number of Steps (N)", 10, 1000, 100, 10)
    pricing_function = binomial_option_pricing
elif selected_model == "Monte Carlo Simulation":
    model_params['num_simulations'] = st.sidebar.slider("Simulations", 1000, 100000, 10000, 1000)
    pricing_function = monte_carlo_option_pricing

@st.cache_data(ttl=3600)
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    hist = stock.history(period="1y")
    if hist.empty:
        raise ValueError("Could not fetch historical data.")
    return info, hist

with st.sidebar.expander("📈 Underlying Stock", expanded=True):
    ticker = st.text_input("Stock Ticker", value="AAPL").upper()
    try:
        info, hist = get_stock_data(ticker)
        spot_price = hist['Close'].iloc[-1]
        company_name = info.get('longName', 'N/A')
        currency = info.get('currency', '$')
        log_returns = np.log(hist['Close'] / hist['Close'].shift(1))
        volatility = np.std(log_returns) * np.sqrt(252)
        # Fetch T-bill rate for risk-free rate
        rf_hist = yf.Ticker("^IRX").history(period="1d")
        risk_free_rate = rf_hist['Close'].iloc[-1] / 100 if not rf_hist.empty else 0.05
        st.success(f"Fetched {company_name}")
    except Exception as e:
        st.warning(f"Could not fetch {ticker} data. Using defaults.")
        spot_price, volatility, risk_free_rate, currency = 150.0, 0.25, 0.05, "$"

    S = st.number_input("Spot Price", value=spot_price, format="%.2f")
    sigma = st.number_input("Volatility (σ)", value=volatility, format="%.4f")
    r = st.number_input("Risk-Free Rate (r)", value=risk_free_rate, format="%.4f")

with st.sidebar.expander("⚙️ Option Contract", expanded=True):
    K = st.number_input("Strike Price", value=spot_price, format="%.2f")
    T_days = st.number_input("Time to Maturity (Days)", value=90, min_value=1)
    T = T_days / 365.0

# --- Function to get pricing and greeks ---
def get_option_data(model, S, K, T, r, sigma, option_type, **kwargs):
    if model == "Black-Scholes":
        price = black_scholes(S, K, T, r, sigma, option_type)
        delta, gamma, theta, vega, rho = bs_greeks(S, K, T, r, sigma, option_type)
    else:
        pricing_func = binomial_option_pricing if model == "Binomial Option Pricing" else monte_carlo_option_pricing
        price = pricing_func(S, K, T, r, sigma, option_type, **kwargs)
        delta, gamma, theta, vega, rho = finite_difference_greeks(pricing_func, S, K, T, r, sigma, option_type, **kwargs)
    return price, delta, gamma, theta, vega, rho

# --- Main Calculation ---
with st.spinner(f"Calculating with {selected_model}..."):
    call_price, cd, cg, ct, cv, cr = get_option_data(selected_model, S, K, T, r, sigma, "call", **model_params)
    put_price, pd, pg, pt, pv, pr = get_option_data(selected_model, S, K, T, r, sigma, "put", **model_params)

# --- Plotly Theme ---
def create_modern_plot_theme():
    return {'layout': {'plot_bgcolor': 'rgba(0,0,0,0)', 'paper_bgcolor': 'rgba(0,0,0,0)', 'font': {'color': '#e0e0e0'}, 'xaxis': {'gridcolor': 'rgba(177, 156, 217, 0.2)'}, 'yaxis': {'gridcolor': 'rgba(177, 156, 217, 0.2)'}, 'legend': {'bgcolor': "rgba(26, 26, 46, 0.7)", 'bordercolor': "rgba(177, 156, 217, 0.5)"}}}

# ------------------- TABS -------------------
tabs = st.tabs(["📋 Summary", "💸 Payoff", "📈 3D Surface", "🔥 Heatmaps", "🎯 Sensitivity"])

with tabs[0]:
    st.header(f"Option Valuation ({selected_model})")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🟢 Call Option")
        st.metric("Price", f"{currency} {call_price:.2f}")
        gcol1, gcol2 = st.columns(2)
        gcol1.metric("Delta (Δ)", f"{cd:.4f}"); gcol2.metric("Gamma (Γ)", f"{cg:.4f}")
        gcol1.metric("Vega", f"{cv:.4f}"); gcol2.metric("Theta (Θ)", f"{ct:.4f}")
        gcol1.metric("Rho (Ρ)", f"{cr:.4f}")
    with col2:
        st.subheader("🔴 Put Option")
        st.metric("Price", f"{currency} {put_price:.2f}")
        gcol1, gcol2 = st.columns(2)
        gcol1.metric("Delta (Δ)", f"{pd:.4f}"); gcol2.metric("Gamma (Γ)", f"{pg:.4f}")
        gcol1.metric("Vega", f"{pv:.4f}"); gcol2.metric("Theta (Θ)", f"{pt:.4f}")
        gcol1.metric("Rho (Ρ)", f"{pr:.4f}")

with tabs[1]:
    st.header("Profit/Loss at Expiration")
    spot_range = np.linspace(S * 0.7, S * 1.3, 100)
    call_payoff = np.maximum(spot_range - K, 0) - call_price
    put_payoff = np.maximum(K - spot_range, 0) - put_price
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spot_range, y=call_payoff, name='Call P/L', line=dict(color='#40E0D0', width=3)))
    fig.add_trace(go.Scatter(x=spot_range, y=put_payoff, name='Put P/L', line=dict(color='#8A2BE2', width=3)))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(224, 224, 224, 0.5)")
    fig.add_vline(x=K, line_dash="dash", line_color="#FF6B6B", annotation_text="Strike Price")
    fig.update_layout(**create_modern_plot_theme()['layout'], title="Option Payoff Profile", xaxis_title="Stock Price at Expiration", yaxis_title="Profit / Loss")
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.header(f"3D Price Surface ({selected_model})")
    @st.cache_data
    def get_3d_data(_model, _S, _K, _T, _r, _sigma, **_kwargs):
        spot_range = np.linspace(0.5*_S, 1.5*_S, 30); time_range = np.linspace(_T, 0.01, 30)
        Spot, Time = np.meshgrid(spot_range, time_range)
        Z = np.zeros_like(Spot)
        for i in range(Spot.shape[0]):
            for j in range(Spot.shape[1]):
                Z[i, j], _, _, _, _, _ = get_option_data(_model, Spot[i, j], _K, Time[i, j], _r, _sigma, "call", **_kwargs)
        return Spot, Time, Z

    Spot_3d, Time_3d, Z_3d = get_3d_data(selected_model, S, K, T, r, sigma, **model_params)
    fig = go.Figure(data=[go.Surface(x=Spot_3d, y=Time_3d, z=Z_3d, colorscale='viridis')])
    fig.update_layout(**create_modern_plot_theme()['layout'], title="Call Option Price vs. Spot & Time", scene = dict(xaxis_title='Spot Price', yaxis_title='Time to Maturity', zaxis_title='Option Price'), margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    st.header(f"Greeks Heatmaps ({selected_model})")
    @st.cache_data
    def get_heatmap_data(_model, _S, _K, _T, _r, _sigma, **_kwargs):
        spot_range = np.linspace(0.8*_S, 1.2*_S, 20); vol_range = np.linspace(max(0.01, _sigma-0.1), _sigma+0.1, 20)
        greeks_data = {greek: np.zeros((len(vol_range), len(spot_range))) for greek in ["Price", "Delta", "Gamma", "Vega", "Theta"]}
        for i, vol in enumerate(vol_range):
            for j, spot in enumerate(spot_range):
                price, delta, gamma, theta, vega, _ = get_option_data(_model, spot, _K, _T, _r, vol, "call", **_kwargs)
                greeks_data["Price"][i, j] = price; greeks_data["Delta"][i, j] = delta; greeks_data["Gamma"][i, j] = gamma; greeks_data["Vega"][i, j] = vega; greeks_data["Theta"][i, j] = theta
        return spot_range, vol_range, greeks_data

    spot_hm, vol_hm, heatmaps_data = get_heatmap_data(selected_model, S, K, T, r, sigma, **model_params)
    
    selected_greek = st.selectbox("Select Greek for Heatmap", ["Price", "Delta", "Gamma", "Vega", "Theta"])
    
    fig = go.Figure(data=go.Heatmap(z=heatmaps_data[selected_greek], x=spot_hm, y=vol_hm, colorscale='viridis', hoverongaps=False))
    fig.update_layout(**create_modern_plot_theme()['layout'], title=f'Call {selected_greek} vs Spot & Volatility', xaxis_title="Spot Price", yaxis_title="Volatility")
    st.plotly_chart(fig, use_container_width=True)

with tabs[4]:
    st.header("Sensitivity Analysis")
    y_axis_value = st.selectbox("Greek to Analyze", ["Price", "Delta", "Gamma", "Theta", "Vega", "Rho"])
    
    @st.cache_data
    def get_sensitivity_data(_model, _S, _K, _T, _r, _sigma, **_kwargs):
        spot_range = np.linspace(0.7*_S, 1.3*_S, 100)
        data = {greek: [] for greek in ["Price", "Delta", "Gamma", "Theta", "Vega", "Rho"]}
        for spot in spot_range:
            price, delta, gamma, theta, vega, rho = get_option_data(_model, spot, _K, _T, _r, _sigma, "call", **_kwargs)
            data["Price"].append(price); data["Delta"].append(delta); data["Gamma"].append(gamma)
            data["Theta"].append(theta); data["Vega"].append(vega); data["Rho"].append(rho)
        return spot_range, data

    spot_cs, cs_data = get_sensitivity_data(selected_model, S, K, T, r, sigma, **model_params)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spot_cs, y=cs_data[y_axis_value], mode='lines', line=dict(width=3, color='#40E0D0')))
    fig.update_layout(**create_modern_plot_theme()['layout'], title=f'Call {y_axis_value} vs. Spot Price', xaxis_title="Spot Price", yaxis_title=y_axis_value)
    st.plotly_chart(fig, use_container_width=True)
