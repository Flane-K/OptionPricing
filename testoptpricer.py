import streamlit as st
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from scipy.stats import norm
import pandas as pd

st.set_page_config(layout="wide", page_title="Option Pricing Visualizer")

# --- Modern Glassmorphic CSS ---
st.markdown("""
<style>
    /* Global dark theme */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        color: #e0e0e0;
    }
    
    /* Main glass container for tab content */
    [data-baseweb="tab-panel"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 25px;
        margin-top: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    /* Sub-container for metrics and text, highlights on hover */
    .sub-container {
        background: transparent;
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid transparent;
        transition: background 0.3s ease, border-color 0.3s ease;
    }

    .sub-container:hover {
        background: rgba(64, 224, 208, 0.1);
        border-color: rgba(64, 224, 208, 0.3);
    }

    /* Custom styling for the manually created metrics inside sub-containers */
    .metric-label {
        color: #B19CD9 !important;
        font-weight: 500;
        font-size: 1rem;
        margin-bottom: 5px;
    }

    .metric-value {
        color: #40E0D0 !important;
        font-size: 1.75rem !important;
        font-weight: 700;
        line-height: 1.2;
        text-shadow: 0 0 8px rgba(64, 224, 208, 0.4);
    }
    
    /* Header styling */
    h1, h2, h3, h4, h5, h6 {
        color: #40E0D0 !important;
        text-shadow: 0 0 10px rgba(64, 224, 208, 0.3);
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

# ------------------- MODEL AND DATA FUNCTIONS (UNCHANGED) -------------------
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type.lower() == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_greeks(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    if option_type.lower() == "call":
        delta = norm.cdf(d1)
        theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1
        theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    return delta, gamma, theta, vega, rho

def binomial_option_pricing(S, K, T, r, sigma, option_type="call", N=100):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    prices = S * (u ** (np.arange(N, -1, -1))) * (d ** (np.arange(0, N + 1, 1)))
    if option_type.lower() == "call":
        option_values = np.maximum(0, prices - K)
    else:
        option_values = np.maximum(0, K - prices)
    for i in range(N - 1, -1, -1):
        option_values = np.exp(-r * dt) * (p * option_values[:-1] + (1 - p) * option_values[1:])
    return option_values[0]

def binomial_greeks(S, K, T, r, sigma, option_type="call", N=100):
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
    price_vol_down = binomial_option_pricing(S, K, T, r, sigma - d_sigma, option_type, N)
    vega = (price_vol_up - price_vol_down) / (2 * d_sigma)
    price_t_down = binomial_option_pricing(S, K, T - dT, r, sigma, option_type, N)
    theta = (price_t_down - price_mid) / dT
    price_r_up = binomial_option_pricing(S, K, T, r + d_r, sigma, option_type, N)
    price_r_down = binomial_option_pricing(S, K, T, r - d_r, sigma, option_type, N)
    rho = (price_r_up - price_r_down) / (2 * d_r)
    return delta, gamma, theta, vega, rho

def monte_carlo_option_pricing(S, K, T, r, sigma, option_type="call", num_simulations=10000):
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * np.random.standard_normal(num_simulations))
    if option_type.lower() == "call":
        payoffs = np.maximum(0, ST - K)
    else:
        payoffs = np.maximum(0, K - ST)
    return np.exp(-r * T) * np.mean(payoffs)

def mc_greeks(S, K, T, r, sigma, option_type="call", num_simulations=10000):
    dS = S * 0.01
    dT = 0.001
    d_sigma = sigma * 0.01
    d_r = r * 0.01 if r > 0 else 0.0001
    price_mid = monte_carlo_option_pricing(S, K, T, r, sigma, option_type, num_simulations)
    price_S_up = monte_carlo_option_pricing(S + dS, K, T, r, sigma, option_type, num_simulations)
    price_S_down = monte_carlo_option_pricing(S - dS, K, T, r, sigma, option_type, num_simulations)
    delta = (price_S_up - price_S_down) / (2 * dS)
    gamma = (price_S_up - 2 * price_mid + price_S_down) / (dS ** 2)
    price_vol_up = monte_carlo_option_pricing(S, K, T, r, sigma + d_sigma, option_type, num_simulations)
    price_vol_down = monte_carlo_option_pricing(S, K, T, r, sigma - d_sigma, option_type, num_simulations)
    vega = (price_vol_up - price_vol_down) / (2 * d_sigma)
    price_t_down = monte_carlo_option_pricing(S, K, T - dT, r, sigma, option_type, num_simulations)
    theta = (price_t_down - price_mid) / dT
    price_r_up = monte_carlo_option_pricing(S, K, T, r + d_r, sigma, option_type, num_simulations)
    price_r_down = monte_carlo_option_pricing(S, K, T, r - d_r, sigma, option_type, num_simulations)
    rho = (price_r_up - price_r_down) / (2 * d_r)
    return delta, gamma, theta, vega, rho

def get_option_value_and_greeks(model, S, K, T, r, sigma, option_type, **kwargs):
    if model == "Black-Scholes":
        price = black_scholes(S, K, T, r, sigma, option_type)
        delta, gamma, theta, vega, rho = bs_greeks(S, K, T, r, sigma, option_type)
    elif model == "Binomial Option Pricing":
        N = kwargs.get('N', 100)
        price = binomial_option_pricing(S, K, T, r, sigma, option_type, N)
        delta, gamma, theta, vega, rho = binomial_greeks(S, K, T, r, sigma, option_type, N)
    else: # Monte Carlo
        num_sims = kwargs.get('num_simulations', 10000)
        price = monte_carlo_option_pricing(S, K, T, r, sigma, option_type, num_sims)
        delta, gamma, theta, vega, rho = mc_greeks(S, K, T, r, sigma, option_type, num_sims)
    return price, delta, gamma, theta, vega, rho

@st.cache_data(ttl=3600)
def get_stock_info(ticker_symbol):
    try: return yf.Ticker(ticker_symbol).info
    except Exception: return {}

@st.cache_data(ttl=3600)
def get_stock_history(ticker_symbol, period):
    try: return yf.Ticker(ticker_symbol).history(period=period)
    except Exception: return pd.DataFrame()
# ------------------- END OF MODEL FUNCTIONS -------------------


# ------------------- Sidebar Controls -------------------
st.sidebar.markdown("## 🔧 Configure Parameters")
selected_model = st.sidebar.selectbox("Select Pricing Model", ["Black-Scholes", "Binomial Option Pricing", "Monte Carlo Simulation"])

if selected_model == "Binomial Option Pricing":
    N_binomial = st.sidebar.slider("Number of Steps (N)", min_value=10, max_value=1000, value=100, step=10)
elif selected_model == "Monte Carlo Simulation":
    num_simulations_mc = st.sidebar.slider("Number of Simulations", min_value=1000, max_value=100000, value=10000, step=1000)

with st.sidebar.expander("📈 Underlying Stock Parameters", expanded=True):
    current_ticker = st.session_state.get('ticker_input', 'AAPL')
    ticker = st.text_input("Enter Stock Ticker", value=current_ticker).upper()
    st.session_state['ticker_input'] = ticker

    info = get_stock_info(ticker)
    company_name = info.get('longName', '').strip()
    st.write(f"**Company Name:** {company_name if company_name else f'Not found for {ticker}.'}")

    spot_price, vol_est, rf_fetch = 100.0, 0.20, 0.03
    currency = "$"
    hist = get_stock_history(ticker, "5d")
    if not hist.empty:
        spot_price = hist["Close"].iloc[-1]
        currency = "₹" if ticker.endswith(".NS") else "$"
        hist30 = get_stock_history(ticker, "30d")["Close"]
        if not hist30.empty:
            log_ret = np.log(hist30 / hist30.shift(1)).dropna()
            vol_est = np.std(log_ret) * np.sqrt(252)

    S = st.number_input("Spot Price", value=float(spot_price), min_value=0.01, format="%.2f")
    sigma = st.number_input("Volatility (σ)", min_value=0.01, max_value=2.0, value=round(vol_est, 2), step=0.01)

    rf_ticker = "^NSITEN" if ticker.endswith(".NS") else "^IRX"
    rf_data = get_stock_history(rf_ticker, "1d")["Close"]
    if not rf_data.empty:
        rf_fetch = rf_data.iloc[-1] / 100
    r = st.number_input("Risk-Free Rate (r)", min_value=0.0, max_value=0.2, value=float(rf_fetch), step=0.001, format="%.3f")

    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

with st.sidebar.expander("⚙️ Option Parameters", expanded=True):
    K = st.number_input("Strike Price", value=float(spot_price), min_value=0.01, format="%.2f")
    T = st.number_input("Time to Maturity (yrs)", min_value=0.01, max_value=5.0, value=0.5, step=0.01)

# ------------------- Main Calculation Block -------------------
model_params = {}
if selected_model == "Binomial Option Pricing":
    model_params['N'] = N_binomial
elif selected_model == "Monte Carlo Simulation":
    model_params['num_simulations'] = num_simulations_mc

with st.spinner(f"🚀 Calculating with {selected_model} model..."):
    call_price, cd, cg, ct, cv, cr = get_option_value_and_greeks(selected_model, S, K, T, r, sigma, "call", **model_params)
    put_price, pd, pg, pt, pv, pr = get_option_value_and_greeks(selected_model, S, K, T, r, sigma, "put", **model_params)

# ------------------- UI HELPER FUNCTIONS -------------------
def create_modern_plot_theme():
    return {
        'layout': go.Layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font={'color': '#e0e0e0', 'family': 'Arial, sans-serif'},
            colorway=['#40E0D0', '#8A2BE2', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
            xaxis={'gridcolor': 'rgba(64, 224, 208, 0.2)', 'zerolinecolor': 'rgba(64, 224, 208, 0.4)', 'color': '#e0e0e0'},
            yaxis={'gridcolor': 'rgba(64, 224, 208, 0.2)', 'zerolinecolor': 'rgba(64, 224, 208, 0.4)', 'color': '#e0e0e0'},
            legend={'bgcolor': "rgba(255,255,255,0.1)", 'bordercolor': "rgba(64, 224, 208, 0.3)", 'borderwidth': 1}
        )
    }

# Helper to create a custom metric card with hover effect
def metric_card(label, value, currency=""):
    return f"""
    <div class="sub-container">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{currency}{value}</div>
    </div>
    """

# ------------------- TABS -------------------
tab_icons = ['📋', '💸', '📊', '📈', '🔥', '🎯']
tab_names = ["Summary", "Payoff Diagram", "Model Comparison", "3D Surface", "Heatmaps", "Cross-Section"]
tabs = st.tabs([f"{icon} {name}" for icon, name in zip(tab_icons, tab_names)])

with tabs[0]:
    st.header(f"Option Valuation ({selected_model})")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(metric_card("🟢 Call Option Price", f"{call_price:.2f}", currency), unsafe_allow_html=True)
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            st.markdown(metric_card("Delta (Δ)", f"{cd:.4f}"), unsafe_allow_html=True)
            st.markdown(metric_card("Vega", f"{cv:.4f}"), unsafe_allow_html=True)
        with gcol2:
            st.markdown(metric_card("Gamma (Γ)", f"{cg:.4f}"), unsafe_allow_html=True)
            st.markdown(metric_card("Theta (Θ)", f"{ct:.4f}"), unsafe_allow_html=True)
        st.markdown(metric_card("Rho (Ρ)", f"{cr:.4f}"), unsafe_allow_html=True)

    with col2:
        st.markdown(metric_card("🔴 Put Option Price", f"{put_price:.2f}", currency), unsafe_allow_html=True)
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            st.markdown(metric_card("Delta (Δ)", f"{pd:.4f}"), unsafe_allow_html=True)
            st.markdown(metric_card("Vega", f"{pv:.4f}"), unsafe_allow_html=True)
        with gcol2:
            st.markdown(metric_card("Gamma (Γ)", f"{pg:.4f}"), unsafe_allow_html=True)
            st.markdown(metric_card("Theta (Θ)", f"{pt:.4f}"), unsafe_allow_html=True)
        st.markdown(metric_card("Rho (Ρ)", f"{pr:.4f}"), unsafe_allow_html=True)

with tabs[1]:
    st.header("Profit/Loss at Expiration")
    with st.container(border=True):
        spot_range = np.linspace(S * 0.7, S * 1.3, 100)
        call_payoff = np.maximum(spot_range - K, 0) - call_price
        put_payoff = np.maximum(K - spot_range, 0) - put_price
        fig = go.Figure(layout=create_modern_plot_theme()['layout'])
        fig.add_trace(go.Scatter(x=spot_range, y=call_payoff, mode='lines', name='Call P/L', line=dict(width=3)))
        fig.add_trace(go.Scatter(x=spot_range, y=put_payoff, mode='lines', name='Put P/L', line=dict(width=3)))
        fig.add_hline(y=0, line_dash="dash", line_color="rgba(224, 224, 224, 0.5)")
        fig.add_vline(x=K, line_dash="dash", line_color="#FF6B6B", annotation_text="Strike")
        fig.update_layout(title="Option Payoff Profile", xaxis_title="Stock Price at Expiration", yaxis_title="Profit / Loss")
        st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.header("Model Price Comparison")
    with st.spinner("Running all models for comparison..."):
        bs_call, bs_cd, _, _, _, _ = get_option_value_and_greeks("Black-Scholes", S, K, T, r, sigma, "call")
        bs_put, bs_pd, _, _, _, _ = get_option_value_and_greeks("Black-Scholes", S, K, T, r, sigma, "put")
        bi_call, bi_cd, _, _, _, _ = get_option_value_and_greeks("Binomial Option Pricing", S, K, T, r, sigma, "call", **model_params)
        bi_put, bi_pd, _, _, _, _ = get_option_value_and_greeks("Binomial Option Pricing", S, K, T, r, sigma, "put", **model_params)
        mc_call, mc_cd, _, _, _, _ = get_option_value_and_greeks("Monte Carlo Simulation", S, K, T, r, sigma, "call", **model_params)
        mc_put, mc_pd, _, _, _, _ = get_option_value_and_greeks("Monte Carlo Simulation", S, K, T, r, sigma, "put", **model_params)

    with st.container(border=True):
        st.subheader("Call Option Comparison")
        st.dataframe(pd.DataFrame({
            "Metric": ["Price", "Delta"],
            "Black-Scholes": [f"{bs_call:.4f}", f"{bs_cd:.4f}"],
            f"Binomial": [f"{bi_call:.4f}", f"{bi_cd:.4f}"],
            f"Monte Carlo": [f"{mc_call:.4f}", f"{mc_cd:.4f}"],
        }), use_container_width=True)

    with st.container(border=True):
        st.subheader("Put Option Comparison")
        st.dataframe(pd.DataFrame({
            "Metric": ["Price", "Delta"],
            "Black-Scholes": [f"{bs_put:.4f}", f"{bs_pd:.4f}"],
            f"Binomial": [f"{bi_put:.4f}", f"{bi_pd:.4f}"],
            f"Monte Carlo": [f"{mc_put:.4f}", f"{mc_pd:.4f}"],
        }), use_container_width=True)

with tabs[3]:
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

    with st.container(border=True):
        Spot, Time, Z_call = get_3d_data("call", selected_model, S, K, T, r, sigma, **model_params)
        fig_call = go.Figure(data=[go.Surface(x=Spot, y=Time, z=Z_call, colorscale='viridis')])
        fig_call.update_layout(**create_modern_plot_theme()['layout'], title="Call Option Price vs. Spot and Time", scene=dict(xaxis_title="Spot", yaxis_title="Time", zaxis_title="Price"))
        st.plotly_chart(fig_call, use_container_width=True)

    with st.container(border=True):
        Spot, Time, Z_put = get_3d_data("put", selected_model, S, K, T, r, sigma, **model_params)
        fig_put = go.Figure(data=[go.Surface(x=Spot, y=Time, z=Z_put, colorscale='plasma')])
        fig_put.update_layout(**create_modern_plot_theme()['layout'], title="Put Option Price vs. Spot and Time", scene=dict(xaxis_title="Spot", yaxis_title="Time", zaxis_title="Price"))
        st.plotly_chart(fig_put, use_container_width=True)

with tabs[4]:
    st.header(f"Price Heatmaps vs. Spot & Volatility ({selected_model})")
    with st.container(border=True):
        @st.cache_data
        def get_heatmap_data(_model, _S_range, _vol_range, _K, _T, _r, **_params):
            call_prices = np.zeros((len(_vol_range), len(_S_range)))
            put_prices = np.zeros((len(_vol_range), len(_S_range)))
            for i, vol in enumerate(_vol_range):
                for j, spot in enumerate(_S_range):
                    call_prices[i, j], _, _, _, _, _ = get_option_value_and_greeks(_model, spot, _K, _T, _r, vol, "call", **_params)
                    put_prices[i, j], _, _, _, _, _ = get_option_value_and_greeks(_model, spot, _K, _T, _r, vol, "put", **_params)
            return call_prices, put_prices

        spot_range_hm = np.linspace(S * 0.8, S * 1.2, 15)
        vol_range_hm = np.linspace(sigma * 0.7, sigma * 1.3, 15)
        call_prices_hm, put_prices_hm = get_heatmap_data(selected_model, spot_range_hm, vol_range_hm, K, T, r, **model_params)
        
        def plot_heatmap(prices, title, colorscale):
            fig = go.Figure(data=go.Heatmap(z=prices, x=spot_range_hm, y=vol_range_hm, colorscale=colorscale))
            fig.update_layout(**create_modern_plot_theme()['layout'], title=title, xaxis_title="Spot Price", yaxis_title="Volatility")
            return fig

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_heatmap(call_prices_hm, "Call Option Prices", "Viridis"), use_container_width=True)
        with col2:
            st.plotly_chart(plot_heatmap(put_prices_hm, "Put Option Prices", "Plasma"), use_container_width=True)
            
with tabs[5]:
    st.header("Sensitivity Analysis")
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        option_type_cs = col1.selectbox("Option Type", ["Call", "Put"], key="opt_type_cs")
        y_axis_value = col2.selectbox("Y-Axis", ["Price", "Delta", "Gamma", "Theta", "Vega", "Rho"], key="y_axis_cs")
        varying_param = col3.selectbox("Varying Param", ["Spot Price", "Time to Maturity"], key="var_param_cs")

        @st.cache_data
        def get_sensitivity_data(_model, _opt_type, _y_axis, _var_param, _S, _K, _T, _r, _sigma, **_params):
            fixed = {"S": _S, "K": _K, "T": _T, "r": _r, "sigma": _sigma}
            param_map = {"Spot Price": "S", "Time to Maturity": "T"}
            var_key = param_map[_var_param]
            low_bound = 0.7 * fixed[var_key]
            if var_key == 'T': low_bound = max(0.01, low_bound)
            x_vals = np.linspace(low_bound, 1.3 * fixed[var_key], 100)
            y_vals = []
            for val in x_vals:
                temp = fixed.copy()
                temp[var_key] = val
                results = get_option_value_and_greeks(_model, temp["S"], temp["K"], temp["T"], temp["r"], temp["sigma"], _opt_type.lower(), **_params)
                greeks_map = {"Price": results[0], "Delta": results[1], "Gamma": results[2], "Theta": results[3], "Vega": results[4], "Rho": results[5]}
                y_vals.append(greeks_map[_y_axis])
            return x_vals, y_vals

        with st.spinner("Generating sensitivity graph..."):
            x_vals_cs, y_vals_cs = get_sensitivity_data(selected_model, option_type_cs, y_axis_value, varying_param, S, K, T, r, sigma, **model_params)
        
        fig = go.Figure(layout=create_modern_plot_theme()['layout'])
        fig.add_trace(go.Scatter(x=x_vals_cs, y=y_vals_cs, mode='lines', line=dict(width=3)))
        fig.update_layout(title=f"{y_axis_value} vs. {varying_param}", xaxis_title=varying_param, yaxis_title=y_axis_value)
        st.plotly_chart(fig, use_container_width=True)
