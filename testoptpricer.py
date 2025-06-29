import streamlit as st
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from scipy.stats import norm
import pandas as pd

st.set_page_config(layout="wide", page_title="Option Pricing Visualizer")

# --- Modern Glassmorphic CSS with Hover Effects ---
st.markdown("""
<style>
    /* Global dark theme */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        color: #e0e0e0;
    }

    /* NEW: Hover-activated sub-window container */
    .hover-container {
        background: transparent;
        border: 1px solid transparent;
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 20px;
        transition: background 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
    }

    .hover-container:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(177, 156, 217, 0.3); /* Purple border on hover */
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Apply glass effect directly to Streamlit's tab content panel */
    [data-baseweb="tab-panel"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 25px;
        margin-top: 15px; /* Add space between tabs and content */
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* MODIFIED: Header styling to purple */
    h1, h2, h3, h4, h5, h6 {
        color: #B19CD9 !important; /* Changed to purple */
        text-shadow: 0 0 10px rgba(177, 156, 217, 0.3); /* Adjusted shadow to match */
        font-weight: 600;
    }
    
    /* Main title - UNCHANGED */
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
    
    /* Metric value styling - UNCHANGED */
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
    
    /* Toggle styling */
    .stToggle > div {
        background: rgba(64, 224, 208, 0.2) !important;
        border-radius: 20px !important;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background: linear-gradient(45deg, #40E0D0, #8A2BE2) !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 10px !important;
        color: #40E0D0 !important;
        font-weight: 600 !important;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-top-color: #40E0D0 !important;
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

# ------------------- Sidebar Controls -------------------
st.sidebar.markdown("## 🔧 Configure Parameters")
selected_model = st.sidebar.selectbox("Select Pricing Model", ["Black-Scholes", "Binomial Option Pricing", "Monte Carlo Simulation"])

if selected_model == "Binomial Option Pricing":
    N_binomial = st.sidebar.slider("Number of Steps (N)", min_value=10, max_value=1000, value=100, step=10)
elif selected_model == "Monte Carlo Simulation":
    num_simulations_mc = st.sidebar.slider("Number of Simulations", min_value=1000, max_value=100000, value=10000, step=1000)

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

    # Fetch company name dynamically and display it directly
    company_name = "N/A"
    info = get_stock_info(ticker)
    fetched_company_name = info.get('longName', '').strip()
    
    if fetched_company_name:
        company_name = fetched_company_name
        st.write(f"**Company Name:** {company_name}")
    else:
        st.write(f"**Company Name:** Not found for '{ticker}'.")

    # Initialize defaults and help texts
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
                vol_help_text = "Could not estimate volatility from 30d history. Using default value."
        else:
            spot_help_text = f"Could not find data for ticker '{ticker}'. Using default value."
            vol_help_text = "Could not estimate volatility. Using default value."
    except Exception as e:
        spot_help_text = f"Error fetching stock data: {e}. Using defaults."
        vol_help_text = "Error fetching volatility. Using default."

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

# ------------------- Function to get pricing and greeks based on selected model -------------------
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

with st.spinner(f"🚀 Calculating with {selected_model} model..."):
    call_price, cd, cg, ct, cv, cr = get_option_value_and_greeks(selected_model, S, K, T, r, sigma, "call", **model_params)
    put_price, pd, pg, pt, pv, pr = get_option_value_and_greeks(selected_model, S, K, T, r, sigma, "put", **model_params)

# ------------------- Enhanced Plotly Theme -------------------
def create_modern_plot_theme():
    return {
        'layout': {
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'font': {'color': '#e0e0e0', 'family': 'Arial, sans-serif'},
            'colorway': ['#40E0D0', '#8A2BE2', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
            'xaxis': {
                'gridcolor': 'rgba(64, 224, 208, 0.2)',
                'zerolinecolor': 'rgba(64, 224, 208, 0.4)',
                'color': '#e0e0e0'
            },
            'yaxis': {
                'gridcolor': 'rgba(64, 224, 208, 0.2)',
                'zerolinecolor': 'rgba(64, 224, 208, 0.4)',
                'color': '#e0e0e0'
            },
             'legend': {
                'bgcolor':"rgba(255,255,255,0.1)",
                'bordercolor': "rgba(64, 224, 208, 0.3)",
                'borderwidth': 1
            }
        }
    }

# ------------------- TABS -------------------
tab_icons = ['📋', '💸', '📊', '📈', '🔥', '🎯']
tab_names = ["Summary", "Payoff Diagram", "Model Comparison", "3D Surface", "Heatmaps", "Cross-Section"]

# Create tabs with icons
tabs = st.tabs([f"{icon} {name}" for icon, name in zip(tab_icons, tab_names)])

# ------------------- Tab 0: Option Summary -------------------
with tabs[0]:
    st.markdown('<div class="hover-container">', unsafe_allow_html=True)
    st.header(f"Option Valuation ({selected_model})")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🟢 Call Option")
        st.metric(label="Price", value=f"{currency} {call_price:.2f}")
        gcol1, gcol2 = st.columns(2)
        gcol1.metric(label="Delta (Δ)", value=f"{cd:.4f}")
        gcol2.metric(label="Gamma (Γ)", value=f"{cg:.4f}")
        gcol1.metric(label="Vega", value=f"{cv:.4f}")
        gcol2.metric(label="Theta (Θ)", value=f"{ct:.4f}")
        gcol1.metric(label="Rho (Ρ)", value=f"{cr:.4f}")

    with col2:
        st.subheader("🔴 Put Option")
        st.metric(label="Price", value=f"{currency} {put_price:.2f}")
        gcol1, gcol2 = st.columns(2)
        gcol1.metric(label="Delta (Δ)", value=f"{pd:.4f}")
        gcol2.metric(label="Gamma (Γ)", value=f"{pg:.4f}")
        gcol1.metric(label="Vega", value=f"{pv:.4f}")
        gcol2.metric(label="Theta (Θ)", value=f"{pt:.4f}")
        gcol1.metric(label="Rho (Ρ)", value=f"{pr:.4f}")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------- Tab 1: Payoff Diagram -------------------
with tabs[1]:
    st.markdown('<div class="hover-container">', unsafe_allow_html=True)
    st.header("Profit/Loss at Expiration")
    
    spot_range = np.linspace(S * 0.7, S * 1.3, 100)
    call_payoff = np.maximum(spot_range - K, 0) - call_price
    put_payoff = np.maximum(K - spot_range, 0) - put_price
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spot_range, y=call_payoff, 
        mode='lines', name='Call Option P/L',
        line=dict(color='#40E0D0', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=spot_range, y=put_payoff, 
        mode='lines', name='Put Option P/L',
        line=dict(color='#8A2BE2', width=3)
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(224, 224, 224, 0.5)")
    fig.add_vline(x=K, line_dash="dash", line_color="#FF6B6B", annotation_text="Strike")

    fig.update_layout(
        **create_modern_plot_theme()['layout'],
        title="Option Payoff Profile",
        xaxis_title="Stock Price at Expiration",
        yaxis_title="Profit / Loss per Share"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------- Tab 2: Model Comparison -------------------
with tabs[2]:
    st.markdown('<div class="hover-container">', unsafe_allow_html=True)
    st.header("Model Price Comparison")
    with st.spinner("Running all models for comparison..."):
        # Black-Scholes
        bs_call, bs_cd, bs_cg, bs_ct, bs_cv, bs_cr = get_option_value_and_greeks("Black-Scholes", S, K, T, r, sigma, "call")
        bs_put, bs_pd, bs_pg, bs_pt, bs_pv, bs_pr = get_option_value_and_greeks("Black-Scholes", S, K, T, r, sigma, "put")

        # Binomial
        n_comp = 100
        if selected_model == "Binomial Option Pricing": n_comp = N_binomial
        bi_call, bi_cd, bi_cg, bi_ct, bi_cv, bi_cr = get_option_value_and_greeks("Binomial Option Pricing", S, K, T, r, sigma, "call", N=n_comp)
        bi_put, bi_pd, bi_pg, bi_pt, bi_pv, bi_pr = get_option_value_and_greeks("Binomial Option Pricing", S, K, T, r, sigma, "put", N=n_comp)

        # Monte Carlo
        sims_comp = 10000
        if selected_model == "Monte Carlo Simulation": sims_comp = num_simulations_mc
        mc_call, mc_cd, mc_cg, mc_ct, mc_cv, mc_cr = get_option_value_and_greeks("Monte Carlo Simulation", S, K, T, r, sigma, "call", num_simulations=sims_comp)
        mc_put, mc_pd, mc_pg, mc_pt, mc_pv, mc_pr = get_option_value_and_greeks("Monte Carlo Simulation", S, K, T, r, sigma, "put", num_simulations=sims_comp)

    st.subheader("Call Option Comparison")
    st.dataframe({
        "Metric": ["Price", "Delta", "Gamma", "Theta", "Vega", "Rho"],
        "Black-Scholes": [f"{bs_call:.4f}", f"{bs_cd:.4f}", f"{bs_cg:.4f}", f"{bs_ct:.4f}", f"{bs_cv:.4f}", f"{bs_cr:.4f}"],
        f"Binomial (N={n_comp})": [f"{bi_call:.4f}", f"{bi_cd:.4f}", f"{bi_cg:.4f}", f"{bi_ct:.4f}", f"{bi_cv:.4f}", f"{bi_cr:.4f}"],
        f"Monte Carlo (Sims={sims_comp})": [f"{mc_call:.4f}", f"{mc_cd:.4f}", f"{mc_cg:.4f}", f"{mc_ct:.4f}", f"{mc_cv:.4f}", f"{mc_cr:.4f}"],
    }, use_container_width=True)
    
    st.subheader("Put Option Comparison")
    st.dataframe({
        "Metric": ["Price", "Delta", "Gamma", "Theta", "Vega", "Rho"],
        "Black-Scholes": [f"{bs_put:.4f}", f"{bs_pd:.4f}", f"{bs_pg:.4f}", f"{bs_pt:.4f}", f"{bs_pv:.4f}", f"{bs_pr:.4f}"],
        f"Binomial (N={n_comp})": [f"{bi_put:.4f}", f"{bi_pd:.4f}", f"{bi_pg:.4f}", f"{bi_pt:.4f}", f"{bi_pv:.4f}", f"{bi_pr:.4f}"],
        f"Monte Carlo (Sims={sims_comp})": [f"{mc_put:.4f}", f"{mc_pd:.4f}", f"{mc_pg:.4f}", f"{mc_pt:.4f}", f"{mc_pv:.4f}", f"{mc_pr:.4f}"],
    }, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------- Tab 3: 3D Graphs -------------------
with tabs[3]:
    st.markdown('<div class="hover-container">', unsafe_allow_html=True)
    st.header(f"3D Price Surface ({selected_model})")
    
    @st.cache_data
    def get_3d_data(option_type, model, _S, _K, _T, _r, _sigma, **kwargs):
        spot_range = np.linspace(0.5*_S, 1.5*_S, 30)
        time_range = np.linspace(_T, 0.01, 30)
        Spot, Time = np.meshgrid(spot_range, time_range)
        Z = np.zeros_like(Spot)

        for i in range(Spot.shape[0]):
            for j in range(Spot.shape[1]):
                Z[i, j], _, _, _, _, _ = get_option_value_and_greeks(model, Spot[i, j], _K, Time[i, j], _r, _sigma, option_type.lower(), **kwargs)
        return Spot, Time, Z

    def plot_3d(option_type, model, **kwargs):
        Spot, Time, Z = get_3d_data(option_type, model, S, K, T, r, sigma, **kwargs)
        fig = go.Figure(data=[go.Surface(x=Spot, y=Time, z=Z, colorscale='viridis', cmin=Z.min(), cmax=Z.max())])
        
        fig.update_layout(
            **create_modern_plot_theme()['layout'],
            title=f"{option_type.capitalize()} Option Price vs. Spot and Time",
            scene=dict(
                xaxis_title="Spot Price", 
                yaxis_title="Time to Maturity", 
                zaxis_title="Option Price",
                xaxis=dict(backgroundcolor="rgba(0, 0, 0, 0)", gridcolor="rgba(64, 224, 208, 0.2)"),
                yaxis=dict(backgroundcolor="rgba(0, 0, 0, 0)", gridcolor="rgba(64, 224, 208, 0.2)"),
                zaxis=dict(backgroundcolor="rgba(0, 0, 0, 0)", gridcolor="rgba(64, 224, 208, 0.2)")
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        return fig

    st.plotly_chart(plot_3d("call", selected_model, **model_params), use_container_width=True)
    st.plotly_chart(plot_3d("put", selected_model, **model_params), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------- Tab 4: Heatmaps -------------------
with tabs[4]:
    st.markdown('<div class="hover-container">', unsafe_allow_html=True)
    st.header(f"Price Heatmaps vs. Spot & Volatility ({selected_model})")
    
    with st.expander("Adjust Heatmap Parameters"):
        min_spot = st.number_input("Min Spot Price", value=round(S * 0.8, 2), key="hm_min_spot")
        max_spot = st.number_input("Max Spot Price", value=round(S * 1.2, 2), key="hm_max_spot")
        min_vol = st.number_input("Min Volatility", value=max(0.01, round(sigma - 0.1, 2)), step=0.01, key="hm_min_vol")
        max_vol = st.number_input("Max Volatility", value=min(1.0, round(sigma + 0.1, 2)), step=0.01, key="hm_max_vol")

    display_values = st.toggle("Display Values on Heatmap", value=True) 

    num_points = 10 
    if not display_values:
        num_points = st.slider("Heatmap Resolution (N x N grid)", min_value=5, max_value=50, value=25, step=5)
    
    @st.cache_data
    def get_heatmap_data(_selected_model, _min_spot, _max_spot, _min_vol, _max_vol, _num_points, _K, _T, _r, **_model_params):
        spot_range = np.linspace(_min_spot, _max_spot, _num_points)
        vol_range = np.linspace(_min_vol, _max_vol, _num_points)
        call_prices = np.zeros((len(vol_range), len(spot_range)))
        put_prices = np.zeros((len(vol_range), len(spot_range)))
        for i, vol in enumerate(vol_range):
            for j, spot in enumerate(spot_range):
                call_prices[i, j], _, _, _, _, _ = get_option_value_and_greeks(_selected_model, spot, _K, _T, _r, vol, "call", **_model_params)
                put_prices[i, j], _, _, _, _, _ = get_option_value_and_greeks(_selected_model, spot, _K, _T, _r, vol, "put", **_model_params)
        return spot_range, vol_range, call_prices, put_prices

    spot_range_hm, vol_range_hm, call_prices_hm, put_prices_hm = get_heatmap_data(selected_model, min_spot, max_spot, min_vol, max_vol, num_points, K, T, r, **model_params)

    def plot_plotly_heatmap(prices, spot_range, vol_range, title, show_values):
        heatmap_trace = go.Heatmap(z=prices, x=spot_range, y=vol_range, hoverongaps=False, colorscale='viridis')
        if show_values: 
            heatmap_trace.text = np.around(prices, 2)
            heatmap_trace.texttemplate = "%{text}"
            
        fig = go.Figure(data=heatmap_trace)
        fig.update_layout(**create_modern_plot_theme()['layout'], title=title, xaxis_title="Spot Price", yaxis_title="Volatility")
        return fig

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_plotly_heatmap(call_prices_hm, spot_range_hm, vol_range_hm, "Call Option Prices", display_values), use_container_width=True)
    with col2:
        st.plotly_chart(plot_plotly_heatmap(put_prices_hm, spot_range_hm, vol_range_hm, "Put Option Prices", display_values), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------- Tab 5: Cross-Section -------------------
with tabs[5]:
    st.markdown('<div class="hover-container">', unsafe_allow_html=True)
    st.header("Sensitivity Analysis")
    col1, col2, col3 = st.columns(3)
    option_type_cs = col1.selectbox("Option Type", ["Call", "Put"], key="opt_type_cs")
    y_axis_value = col2.selectbox("Y-Axis Value", ["Price", "Delta", "Gamma", "Theta", "Vega", "Rho"], key="y_axis_cs")
    varying_param = col3.selectbox("Parameter to Vary", ["Spot Price", "Strike Price", "Volatility", "Time to Maturity"], key="var_param_cs")
    
    @st.cache_data
    def get_sensitivity_data(_selected_model, _option_type, _y_axis, _varying_param, _S, _K, _T, _r, _sigma, **_model_params):
        fixed = {"S": _S, "K": _K, "T": _T, "r": _r, "sigma": _sigma}
        param_map = {"Spot Price": "S", "Strike Price": "K", "Volatility": "sigma", "Time to Maturity": "T"}
        var_param_key = param_map[_varying_param]

        # Ensure the range is valid, especially for T and sigma which cannot be zero or negative
        low_bound = 0.7 * fixed[var_param_key]
        if var_param_key in ['T', 'sigma']:
            low_bound = max(0.01, low_bound)
            
        x_vals = np.linspace(low_bound, 1.3 * fixed[var_param_key], 100)
        y_vals = []
        
        for val in x_vals:
            temp = fixed.copy()
            temp[var_param_key] = val
            price, delta, gamma, theta, vega, rho = get_option_value_and_greeks(_selected_model, temp["S"], temp["K"], temp["T"], temp["r"], temp["sigma"], _option_type.lower(), **_model_params)
            greeks_map = {"Price": price, "Delta": delta, "Gamma": gamma, "Theta": theta, "Vega": vega, "Rho": rho}
            y_vals.append(greeks_map[_y_axis])
        return x_vals, y_vals
    
    with st.spinner("Generating sensitivity graph..."):
        x_vals_cs, y_vals_cs = get_sensitivity_data(selected_model, option_type_cs, y_axis_value, varying_param, S, K, T, r, sigma, **model_params)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals_cs, y=y_vals_cs, mode='lines', line=dict(width=3)))
    fig.update_layout(
        **create_modern_plot_theme()['layout'],
        title=f"{option_type_cs} {y_axis_value} vs. {varying_param} ({selected_model})",
        xaxis_title=varying_param,
        yaxis_title=y_axis_value
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
