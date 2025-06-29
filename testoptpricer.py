import streamlit as st
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from scipy.stats import norm
import pandas as pd

st.set_page_config(layout="wide", page_title="Option Pricing Visualizer", initial_sidebar_state="expanded")

# --- Enhanced Glassmorphic CSS Styling ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 25%, #16213e 50%, #0f3460 75%, #0a0a0a 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Glassmorphic container base */
    .glass-container {
        background: rgba(15, 25, 35, 0.25);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(139, 69, 255, 0.2);
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .glass-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(139, 69, 255, 0.4), transparent);
    }
    
    .glass-container:hover {
        border: 1px solid rgba(139, 69, 255, 0.6);
        box-shadow: 
            0 12px 40px rgba(139, 69, 255, 0.15),
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.15);
        transform: translateY(-2px);
    }
    
    /* Title styling */
    .main-title {
        background: linear-gradient(135deg, #64b5f6 0%, #bb86fc 50%, #03dac6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 0 30px rgba(139, 69, 255, 0.3);
    }
    
    /* Section headers */
    .section-header {
        color: #bb86fc;
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-shadow: 0 0 20px rgba(187, 134, 252, 0.4);
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #03dac6;
        font-size: 2rem;
        font-weight: 600;
        text-shadow: 0 0 15px rgba(3, 218, 198, 0.4);
    }
    
    [data-testid="stMetricLabel"] {
        color: #e1e1e1;
        font-weight: 500;
    }
    
    /* Input styling */
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select {
        background: rgba(25, 35, 45, 0.4) !important;
        border: 1px solid rgba(139, 69, 255, 0.3) !important;
        border-radius: 12px !important;
        color: #e1e1e1 !important;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border: 1px solid rgba(139, 69, 255, 0.8) !important;
        box-shadow: 0 0 20px rgba(139, 69, 255, 0.2) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #8b45ff 0%, #03dac6 100%);
        border: none;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(139, 69, 255, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(139, 69, 255, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(10, 10, 26, 0.8);
        backdrop-filter: blur(20px);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(25, 35, 45, 0.3);
        border-radius: 15px;
        padding: 5px;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        color: #a0a0a0;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #8b45ff 0%, #03dac6 100%);
        color: white;
    }
    
    /* DataFrame styling */
    .stDataFrame {
        background: rgba(15, 25, 35, 0.3);
        border-radius: 15px;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(139, 69, 255, 0.2);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(25, 35, 45, 0.4);
        border-radius: 12px;
        border: 1px solid rgba(139, 69, 255, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* General text styling */
    .stMarkdown p, .stMarkdown li {
        color: #e1e1e1;
        line-height: 1.6;
    }
    
    /* Strong text highlighting */
    strong {
        color: #03dac6;
        font-weight: 600;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-color: #bb86fc transparent #03dac6 transparent;
    }
    
    /* Custom glass metric container */
    .metric-container {
        background: rgba(15, 25, 35, 0.3);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(139, 69, 255, 0.2);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        border: 1px solid rgba(139, 69, 255, 0.5);
        box-shadow: 0 8px 25px rgba(139, 69, 255, 0.1);
    }
    
    /* Loading animation */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .loading {
        animation: pulse 2s infinite;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(25, 35, 45, 0.3);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #8b45ff, #03dac6);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #a055ff, #05e6d0);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Custom function to wrap content in glass containers
def glass_container(content):
    st.markdown(f'<div class="glass-container">{content}</div>', unsafe_allow_html=True)

# Main title with glassmorphic styling
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

    # Delta
    price_S_up = binomial_option_pricing(S + dS, K, T, r, sigma, option_type, N)
    price_S_down = binomial_option_pricing(S - dS, K, T, r, sigma, option_type, N)
    delta = (price_S_up - price_S_down) / (2 * dS)

    # Gamma
    gamma = (price_S_up - 2 * price_mid + price_S_down) / (dS ** 2)

    # Vega
    price_vol_up = binomial_option_pricing(S, K, T, r, sigma + d_sigma, option_type, N)
    price_vol_down = binomial_option_pricing(S, K, T, r, sigma - d_sigma, option_type, N)
    vega = (price_vol_up - price_vol_down) / (2 * d_sigma)

    # Theta
    price_t_down = binomial_option_pricing(S, K, T - dT, r, sigma, option_type, N)
    theta = (price_t_down - price_mid) / dT

    # Rho
    price_r_up = binomial_option_pricing(S, K, T, r + d_r, sigma, option_type, N)
    price_r_down = binomial_option_pricing(S, K, T, r - d_r, sigma, option_type, N)
    rho = (price_r_up - price_r_down) / (2 * d_r)

    return delta, gamma, theta, vega, rho

# ------------------- Monte Carlo Simulation Model -------------------
def monte_carlo_option_pricing(S, K, T, r, sigma, option_type="call", num_simulations=10000):
    """Calculates option price using Monte Carlo simulation."""
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * np.random.standard_normal(num_simulations))
    
    if option_type.lower() == "call":
        payoffs = np.maximum(0, ST - K)
    else: # Put
        payoffs = np.maximum(0, K - ST)
        
    return np.exp(-r * T) * np.mean(payoffs)

def mc_greeks(S, K, T, r, sigma, option_type="call", num_simulations=10000):
    """Calculates Greeks using Monte Carlo (finite differences)."""
    dS = S * 0.01
    dT = 0.001
    d_sigma = sigma * 0.01
    d_r = r * 0.01 if r > 0 else 0.0001
    
    price_mid = monte_carlo_option_pricing(S, K, T, r, sigma, option_type, num_simulations)

    # Delta
    price_S_up = monte_carlo_option_pricing(S + dS, K, T, r, sigma, option_type, num_simulations)
    price_S_down = monte_carlo_option_pricing(S - dS, K, T, r, sigma, option_type, num_simulations)
    delta = (price_S_up - price_S_down) / (2 * dS)

    # Gamma
    gamma = (price_S_up - 2 * price_mid + price_S_down) / (dS ** 2)

    # Vega
    price_vol_up = monte_carlo_option_pricing(S, K, T, r, sigma + d_sigma, option_type, num_simulations)
    price_vol_down = monte_carlo_option_pricing(S, K, T, r, sigma - d_sigma, option_type, num_simulations)
    vega = (price_vol_up - price_vol_down) / (2 * d_sigma)

    # Theta
    price_t_down = monte_carlo_option_pricing(S, K, T - dT, r, sigma, option_type, num_simulations)
    theta = (price_t_down - price_mid) / dT

    # Rho
    price_r_up = monte_carlo_option_pricing(S, K, T, r + d_r, sigma, option_type, num_simulations)
    price_r_down = monte_carlo_option_pricing(S, K, T, r - d_r, sigma, option_type, num_simulations)
    rho = (price_r_up - price_r_down) / (2 * d_r)

    return delta, gamma, theta, vega, rho

# ------------------- Enhanced Plot Styling -------------------
def style_plotly_chart(fig, title="", dark_theme=True):
    """Apply consistent dark glassmorphic styling to plotly charts"""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, color='#bb86fc', family='Inter'),
            x=0.5
        ),
        paper_bgcolor='rgba(15, 25, 35, 0.1)',
        plot_bgcolor='rgba(25, 35, 45, 0.2)',
        font=dict(color='#e1e1e1', family='Inter'),
        xaxis=dict(
            gridcolor='rgba(139, 69, 255, 0.2)',
            linecolor='rgba(139, 69, 255, 0.4)',
            titlefont=dict(color='#03dac6'),
            tickfont=dict(color='#e1e1e1')
        ),
        yaxis=dict(
            gridcolor='rgba(139, 69, 255, 0.2)',
            linecolor='rgba(139, 69, 255, 0.4)',
            titlefont=dict(color='#03dac6'),
            tickfont=dict(color='#e1e1e1')
        ),
        legend=dict(
            bgcolor='rgba(25, 35, 45, 0.5)',
            bordercolor='rgba(139, 69, 255, 0.3)',
            borderwidth=1,
            font=dict(color='#e1e1e1')
        ),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig

# ------------------- Sidebar Controls -------------------
st.sidebar.markdown('<h2 class="section-header">🔧 Configure Parameters</h2>', unsafe_allow_html=True)

with st.sidebar:
    selected_model = st.selectbox("Select Pricing Model", ["Black-Scholes", "Binomial Option Pricing", "Monte Carlo Simulation"])

    if selected_model == "Binomial Option Pricing":
        N_binomial = st.slider("Number of Steps (N)", min_value=10, max_value=1000, value=100, step=10)
    elif selected_model == "Monte Carlo Simulation":
        num_simulations_mc = st.slider("Number of Simulations", min_value=1000, max_value=100000, value=10000, step=1000)

# Caching yfinance info to speed up fetches
@st.cache_data(ttl=3600)
def get_stock_info(ticker_symbol):
    try:
        stock_data = yf.Ticker(ticker_symbol)
        return stock_data.info
    except Exception:
        return {}

@st.cache_data(ttl=3600)
def get_stock_history(ticker_symbol, period):
    try:
        stock_data = yf.Ticker(ticker_symbol)
        return stock_data.history(period=period)
    except Exception:
        return pd.DataFrame()

with st.sidebar.expander("📈 Underlying Stock Parameters", expanded=True):
    current_ticker = st.session_state.get('ticker_input', 'AAPL')
    ticker = st.text_input("Enter Stock Ticker", value=current_ticker).upper()
    st.session_state['ticker_input'] = ticker

    # Fetch company name dynamically
    company_name = "N/A"
    info = get_stock_info(ticker)
    fetched_company_name = info.get('longName', '').strip()
    
    if fetched_company_name:
        company_name = fetched_company_name
        st.markdown(f"**Company:** {company_name}")
    else:
        st.markdown(f"**Company:** Not found for '{ticker}'")

    # Initialize defaults
    spot_price, vol_est, rf_fetch = 100.0, 0.20, 0.03
    spot_help_text = "Default value is 100.00. Enter a ticker to fetch live data."
    vol_help_text = "Default value is 20%. Volatility is estimated from the last 30 days of historical data."
    rf_help_text = "Default value is 3%. Risk-free rate is fetched based on the stock's market."
    currency = "$"
    
    try:
        hist = get_stock_history(ticker, "5d")
        if not hist.empty:
            spot_price = hist["Close"].iloc[-1]
            currency = "₹" if ticker.endswith(".NS") else "$"
            spot_help_text = f"Successfully fetched Spot Price: {currency}{spot_price:.2f}"

            hist30 = get_stock_history(ticker, "30d")["Close"]
            if not hist30.empty:
                log_ret = np.log(hist30 / hist30.shift(1)).dropna()
                vol_est = np.std(log_ret) * np.sqrt(252)
                vol_help_text = f"Estimated Volatility (30d Ann.): {vol_est:.2%}"
        else:
            spot_help_text = f"Could not find data for ticker '{ticker}'. Using default value."
    except Exception as e:
        spot_help_text = f"Error fetching stock data: {e}. Using defaults."

    S = st.number_input("Spot Price", value=float(spot_price), min_value=0.01, format="%.2f", help=spot_help_text)
    sigma = st.number_input("Volatility (σ)", min_value=0.01, max_value=2.0, value=round(vol_est, 2), step=0.01, help=vol_help_text)

    # Dynamic Risk-Free Rate Fetching
    if ticker.endswith(".NS"):
        rf_ticker, rf_name = "^NSITEN", "India 10Y Bond"
    else:
        rf_ticker, rf_name = "^IRX", "US 13W T-Bill"

    try:
        rf_data = get_stock_history(rf_ticker, "1d")["Close"]
        if not rf_data.empty:
            rf_fetch = rf_data.iloc[-1] / 100
            rf_help_text = f"Fetched {rf_name} rate: {rf_fetch:.3%}"
        else:
            rf_help_text = f"Could not fetch {rf_name} rate. Using default."
    except Exception:
        rf_help_text = f"Error fetching {rf_name} rate. Using default."

    r = st.number_input("Risk-Free Rate (r)", min_value=0.0, max_value=0.2, value=float(rf_fetch), step=0.001, format="%.3f", help=rf_help_text)

    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        try:
            st.rerun()
        except AttributeError:
            pass

with st.sidebar.expander("⚙️ Option Parameters", expanded=True):
    K = st.number_input("Strike Price", value=float(spot_price), min_value=0.01, format="%.2f")
    T = st.number_input("Time to Maturity (yrs)", min_value=0.01, max_value=5.0, value=0.5, step=0.01)

# ------------------- Function to get pricing and greeks -------------------
def get_option_value_and_greeks(model, S, K, T, r, sigma, option_type, **kwargs):
    if model == "Black-Scholes":
        price = black_scholes(S, K, T, r, sigma, option_type)
        delta, gamma, theta, vega, rho = bs_greeks(S, K, T, r, sigma, option_type)
    elif model == "Binomial Option Pricing":
        N = kwargs.get('N', 100)
        price = binomial_option_pricing(S, K, T, r, sigma, option_type, N)
        delta, gamma, theta, vega, rho = binomial_greeks(S, K, T, r, sigma, option_type, N)
    elif model == "Monte Carlo Simulation":
        num_sims = kwargs.get('num_simulations', 10000)
        price = monte_carlo_option_pricing(S, K, T, r, sigma, option_type, num_sims)
        delta, gamma, theta, vega, rho = mc_greeks(S, K, T, r, sigma, option_type, num_sims)
    
    return price, delta, gamma, theta, vega, rho

# ------------------- Main Calculation Block -------------------
model_params = {}
if selected_model == "Binomial Option Pricing":
    model_params['N'] = N_binomial
elif selected_model == "Monte Carlo Simulation":
    model_params['num_simulations'] = num_simulations_mc

with st.spinner(f"Calculating with {selected_model} model, please wait..."):
    call_price, cd, cg, ct, cv, cr = get_option_value_and_greeks(selected_model, S, K, T, r, sigma, "call", **model_params)
    put_price, pd, pg, pt, pv, pr = get_option_value_and_greeks(selected_model, S, K, T, r, sigma, "put", **model_params)

# ------------------- TABS -------------------
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Summary", "💸 Payoff Diagram", "📊 Model Comparison", "📈 3D Surface", "🔥 Heatmaps", "🎯 Cross-Section"
])

# ------------------- Tab 0: Option Summary -------------------
with tab0:
    st.markdown(f'<h2 class="section-header">Option Valuation ({selected_model})</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">📞 Call Option</h3>', unsafe_allow_html=True)
        st.metric(label="Price", value=f"{currency} {call_price:.2f}")
        
        gcol1, gcol2 = st.columns(2)
        gcol1.metric(label="Delta (Δ)", value=f"{cd:.4f}")
        gcol2.metric(label="Gamma (Γ)", value=f"{cg:.4f}")
        gcol1.metric(label="Vega", value=f"{cv:.4f}")
        gcol2.metric(label="Theta (Θ)", value=f"{ct:.4f}")
        gcol1.metric(label="Rho (Ρ)", value=f"{cr:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">📉 Put Option</h3>', unsafe_allow_html=True)
        st.metric(label="Price", value=f"{currency} {put_price:.2f}")
        
        gcol1, gcol2 = st.columns(2)
        gcol1.metric(label="Delta (Δ)", value=f"{pd:.4f}")
        gcol2.metric(label="Gamma (Γ)", value=f"{pg:.4f}")
        gcol1.metric(label="Vega", value=f"{pv:.4f}")
        gcol2.metric(label="Theta (Θ)", value=f"{pt:.4f}")
        gcol1.metric(label="Rho (Ρ)", value=f"{pr:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)

# ------------------- Tab 1: Payoff Diagram -------------------
with tab1:
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">💰 Profit/Loss at Expiration</h2>', unsafe_allow_html=True)
    
    spot_range = np.linspace(S * 0.7, S * 1.3, 100)
    call_payoff = np.maximum(spot_range - K, 0) - call_price
    put_payoff = np.maximum(K - spot_range, 0) - put_price
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spot_range, y=call_payoff, 
        mode='lines', name='Call Option P/L',
        line=dict(color='#03dac6', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=spot_range, y=put_payoff, 
        mode='lines', name='Put Option P/L',
        line=dict(color='#bb86fc',
