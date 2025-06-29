import streamlit as st
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from scipy.stats import norm
import pandas as pd

st.set_page_config(layout="wide", page_title="Option Pricing Visualizer")

# --- Modern Glassmorphic CSS with Corrected Hover Effect ---
st.markdown("""
<style>
    /* Global dark theme */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        color: #e0e0e0;
    }
    
    /* CORRECTED: Apply hover effect to the entire tab panel */
    [data-baseweb="tab-panel"] {
        background: transparent;
        border: 1px solid transparent;
        border-radius: 20px;
        padding: 25px;
        margin-top: 15px; /* Space between tabs and content */
        transition: background 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
    }

    [data-baseweb="tab-panel"]:hover {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-color: rgba(177, 156, 217, 0.4); /* Purple border on hover */
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* MODIFIED: Header styling to purple */
    h1, h2, h3, h4, h5, h6 {
        color: #B19CD9 !important; /* Changed to purple */
        text-shadow: 0 0 10px rgba(177, 156, 217, 0.3); /* Adjusted shadow to match */
        font-weight: 600;
    }
    
    /* Main title */
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
        color: #B19CD9 !important; /* Label in purple to match headings */
        font-weight: 500;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(26, 26, 46, 0.8) !important;
        backdrop-filter: blur(20px);
    }
    
    /* Tab styling */
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
    
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus {
        border-color: #40E0D0 !important;
        box-shadow: 0 0 15px rgba(64, 224, 208, 0.3) !important;
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
        letter-spacing: 1px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(64, 224, 208, 0.4) !important;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 15px;
        overflow: hidden;
        backdrop-filter: blur(10px);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 10px !important;
        color: #40E0D0 !important;
        font-weight: 600 !important;
    }
    
</style>
""", unsafe_allow_html=True)

# Title with modern styling
st.markdown('<h1 class="main-title">📈 Option Pricing Visualizer</h1>', unsafe_allow_html=True)

# ------------------- Black-Scholes Model -------------------
def black_scholes(S, K, T, r, sigma, option_type="call"):
    """Calculates the Black-Scholes option price."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type.lower() == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # Put
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_greeks(S, K, T, r, sigma, option_type="call"):
    """Calculates the Greeks for the Black-Scholes model."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    
    if option_type.lower() == "call":
        delta = norm.cdf(d1)
        theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:  # Put
        delta = norm.cdf(d1) - 1
        theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        
    return delta, gamma, theta, vega, rho

# ------------------- Binomial Option Pricing Model -------------------
def binomial_option_pricing(S, K, T, r, sigma, option_type="call", N=100):
    """Calculates option price using the Cox-Ross-Rubinstein binomial model."""
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    prices = S * (u ** (np.arange(N, -1, -1))) * (d ** (np.arange(0, N + 1, 1)))
    
    if option_type.lower() == "call":
        option_values = np.maximum(0, prices - K)
    else: # Put
        option_values = np.maximum(0, K - prices)

    for i in range(N - 1, -1, -1):
        option_values = np.exp(-r * dt) * (p * option_values[:-1] + (1 - p) * option_values[1:])
    
    return option_values[0]

def binomial_greeks(S, K, T, r, sigma, option_type="call", N=100):
    """Calculates Greeks using the binomial model (finite differences)."""
    dS = S * 0.01
    dT = T / N if T > 0 else 0.001
    d_sigma = sigma * 0.01
    d_r = r * 0.01 if r > 0 else 0.0001
    
    price_mid = binomial_option_pricing(S, K, T, r, sigma, option_type, N)
    price_S_up = binomial_option_pricing(S + dS, K, T, r, sigma, option_type, N)
    price_S_down = binomial_option_pricing(S - dS, K, T, r, sigma, option_type, N)
    delta = (price_S_up - price_S_down) / (2 * dS)
    gamma = (price_S_up - 2 * price_mid + price_S_down) / (dS ** 2)
    price_vol_up = binomial_option_pricing(S, K, T, r, sigma + d_sigma, option_type, N)
    vega = (price_vol_up - price_mid) / d_sigma
    price_t_down = binomial_option_pricing(S, K, T - dT, r, sigma, option_type, N)
    theta = (price_t_down - price_mid) / dT
    price_r_up = binomial_option_pricing(S, K, T, r + d_r, sigma, option_type, N)
    rho = (price_r_up - price_mid) / d_r
    return delta, gamma, theta, vega, rho

# ------------------- Monte Carlo Simulation Model -------------------
def monte_carlo_option_pricing(S, K, T, r, sigma, option_type="call", num_simulations=10000):
    """Calculates option price using Monte Carlo simulation."""
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * np.random.standard_normal(num_simulations))
    payoffs = np.maximum(0, ST - K) if option_type.lower() == "call" else np.maximum(0, K - ST)
    return np.exp(-r * T) * np.mean(payoffs)

def mc_greeks(S, K, T, r, sigma, option_type="call", num_simulations=10000):
    """Calculates Greeks using Monte Carlo (finite differences)."""
    dS = S * 0.01
    d_sigma = sigma * 0.01
    price_mid = monte_carlo_option_pricing(S, K, T, r, sigma, option_type, num_simulations)
    price_S_up = monte_carlo_option_pricing(S + dS, K, T, r, sigma, option_type, num_simulations)
    price_S_down = monte_carlo_option_pricing(S - dS, K, T, r, sigma, option_type, num_simulations)
    delta = (price_S_up - price_S_down) / (2 * dS)
    gamma = (price_S_up - 2 * price_mid + price_S_down) / (dS ** 2)
    price_vol_up = monte_carlo_option_pricing(S, K, T, r, sigma + d_sigma, option_type, num_simulations)
    vega = (price_vol_up - price_mid) / d_sigma
    price_t_down = monte_carlo_option_pricing(S, K, T - 0.001, r, sigma, option_type, num_simulations)
    theta = (price_t_down - price_mid) / 0.001
    price_r_up = monte_carlo_option_pricing(S, K, T, r + 0.0001, sigma, option_type, num_simulations)
    rho = (price_r_up - price_mid) / 0.0001
    return delta, gamma, theta, vega, rho

# ------------------- Sidebar Controls -------------------
st.sidebar.markdown("## 🔧 Configure Parameters")
selected_model = st.sidebar.selectbox("Select Pricing Model", ["Black-Scholes", "Binomial Option Pricing", "Monte Carlo Simulation"])
model_params = {}
if selected_model == "Binomial Option Pricing":
    model_params['N'] = st.sidebar.slider("Number of Steps (N)", 10, 1000, 100, 10)
elif selected_model == "Monte Carlo Simulation":
    model_params['num_simulations'] = st.sidebar.slider("Number of Simulations", 1000, 100000, 10000, 1000)

@st.cache_data(ttl=3600)
def get_stock_info(ticker):
    return yf.Ticker(ticker).info

@st.cache_data(ttl=3600)
def get_stock_history(ticker, period="1y"):
    return yf.Ticker(ticker).history(period=period)

with st.sidebar.expander("📈 Underlying Stock Parameters", expanded=True):
    ticker = st.text_input("Enter Stock Ticker", value="AAPL").upper()
    try:
        info = get_stock_info(ticker)
        hist = get_stock_history(ticker, period="1y")
        spot_price = hist['Close'].iloc[-1]
        company_name = info.get('longName', 'N/A')
        currency = info.get('currency', '$')
        log_returns = np.log(hist['Close'] / hist['Close'].shift(1))
        volatility = np.std(log_returns) * np.sqrt(252)
        rf_ticker = "^IRX" if info.get('market') == 'us_market' else "^TNX" # Basic logic
        rf_hist = get_stock_history(rf_ticker, period="1d")
        risk_free_rate = rf_hist['Close'].iloc[-1] / 100
        st.success(f"Fetched data for {company_name}")
    except Exception as e:
        st.warning(f"Could not fetch live data for {ticker}. Using defaults. Error: {e}")
        spot_price, volatility, risk_free_rate, currency = 100.0, 0.2, 0.01, "$"

    S = st.number_input("Spot Price", value=spot_price, format="%.2f")
    sigma = st.number_input("Volatility (σ)", value=volatility, format="%.4f")
    r = st.number_input("Risk-Free Rate (r)", value=risk_free_rate, format="%.4f")

with st.sidebar.expander("⚙️ Option Parameters", expanded=True):
    K = st.number_input("Strike Price", value=spot_price, format="%.2f")
    T = st.number_input("Time to Maturity (yrs)", value=1.0, step=0.01)

# ------------------- Calculation & Plotting Functions -------------------
def get_option_value_and_greeks(model, S, K, T, r, sigma, option_type, **kwargs):
    if model == "Black-Scholes":
        price = black_scholes(S, K, T, r, sigma, option_type)
        delta, gamma, theta, vega, rho = bs_greeks(S, K, T, r, sigma, option_type)
    elif model == "Binomial Option Pricing":
        price = binomial_option_pricing(S, K, T, r, sigma, option_type, **kwargs)
        delta, gamma, theta, vega, rho = binomial_greeks(S, K, T, r, sigma, option_type, **kwargs)
    else: # Monte Carlo
        price = monte_carlo_option_pricing(S, K, T, r, sigma, option_type, **kwargs)
        delta, gamma, theta, vega, rho = mc_greeks(S, K, T, r, sigma, option_type, **kwargs)
    return price, delta, gamma, theta, vega, rho

def create_modern_plot_theme():
    return {'layout': {'plot_bgcolor': 'rgba(0,0,0,0)', 'paper_bgcolor': 'rgba(0,0,0,0)', 'font': {'color': '#e0e0e0'}, 'xaxis': {'gridcolor': 'rgba(64, 224, 208, 0.2)'}, 'yaxis': {'gridcolor': 'rgba(64, 224, 208, 0.2)'}, 'legend': {'bgcolor': "rgba(255,255,255,0.1)", 'bordercolor': "rgba(64, 224, 208, 0.3)"}}}

with st.spinner(f"🚀 Calculating with {selected_model} model..."):
    call_price, cd, cg, ct, cv, cr = get_option_value_and_greeks(selected_model, S, K, T, r, sigma, "call", **model_params)
    put_price, pd, pg, pt, pv, pr = get_option_value_and_greeks(selected_model, S, K, T, r, sigma, "put", **model_params)

# ------------------- TABS -------------------
tabs = st.tabs(["📋 Summary", "💸 Payoff", "📊 Comparison", "📈 3D Surface", "🔥 Heatmaps", "🎯 Sensitivity"])

with tabs[0]:
    st.header(f"Option Valuation ({selected_model})")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🟢 Call Option")
        st.metric(label="Price", value=f"{currency} {call_price:.2f}")
        gcol1, gcol2 = st.columns(2)
        gcol1.metric("Delta (Δ)", f"{cd:.4f}"); gcol2.metric("Gamma (Γ)", f"{cg:.4f}")
        gcol1.metric("Vega", f"{cv:.4f}"); gcol2.metric("Theta (Θ)", f"{ct:.4f}")
        gcol1.metric("Rho (Ρ)", f"{cr:.4f}")
    with col2:
        st.subheader("🔴 Put Option")
        st.metric(label="Price", value=f"{currency} {put_price:.2f}")
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
    fig.add_vline(x=K, line_dash="dash", line_color="#FF6B6B", annotation_text="Strike")
    fig.update_layout(**create_modern_plot_theme()['layout'], title="Option Payoff Profile", xaxis_title="Stock Price", yaxis_title="Profit / Loss")
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.header("Model Price Comparison")
    # ... (comparison logic from original script)
    st.info("Model comparison results will be displayed here.")


with tabs[3]:
    st.header(f"3D Price Surface ({selected_model})")
    @st.cache_data
    def get_3d_data(option_type, _S, _T):
        spot_range = np.linspace(0.5*_S, 1.5*_S, 30); time_range = np.linspace(_T, 0.01, 30)
        Spot, Time = np.meshgrid(spot_range, time_range)
        Z = np.array([black_scholes(s, K, t, r, sigma, option_type) for s, t in zip(np.ravel(Spot), np.ravel(Time))]).reshape(Spot.shape)
        return Spot, Time, Z
    Spot_3d, Time_3d, Z_3d = get_3d_data("call", S, T)
    fig = go.Figure(data=[go.Surface(x=Spot_3d, y=Time_3d, z=Z_3d, colorscale='viridis')])
    fig.update_layout(**create_modern_plot_theme()['layout'], title="Call Price vs. Spot & Time", scene = dict(xaxis_title='Spot Price', yaxis_title='Time', zaxis_title='Option Price'), margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig, use_container_width=True)


with tabs[4]:
    st.header("Price Heatmaps")
    # ... (heatmap logic from original script)
    st.info("Heatmaps will be displayed here.")


with tabs[5]:
    st.header("Sensitivity Analysis")
    # ... (sensitivity logic from original script)
    st.info("Sensitivity graphs will be displayed here.")
