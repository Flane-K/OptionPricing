import streamlit as st
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from scipy.stats import norm
import pandas as pd

st.set_page_config(layout="wide", page_title="Option Pricing Visualizer")

# --- Custom CSS for Glassmorphism and Modern Dark Theme ---
st.markdown(
    """
    <style>
    /* ------------------- General Theme & Background ------------------- */
    body {
        color: #E0E0E0; /* Light grey text for readability */
    }

    /* Set the main background for the app */
    .stApp {
        background-color: #0E002B; /* Deep purple-black background */
        background-image:
            radial-gradient(at 20% 25%, hsla(271, 95%, 28%, 0.3) 0px, transparent 50%),
            radial-gradient(at 80% 85%, hsla(212, 95%, 38%, 0.3) 0px, transparent 50%),
            radial-gradient(at 50% 50%, hsla(300, 95%, 15%, 0.4) 0px, transparent 70%);
        background-attachment: fixed;
    }

    /* ------------------- Glassmorphic Containers ------------------- */
    /* Main content area */
    .main .block-container {
        background: rgba(15, 12, 41, 0.5); /* Semi-transparent dark base */
        border: 1px solid rgba(138, 43, 226, 0.3); /* Subtle purple border */
        backdrop-filter: blur(15px); /* The frosted glass effect */
        -webkit-backdrop-filter: blur(15px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37); /* Soft shadow for depth */
        border-radius: 15px; /* Rounded corners */
        padding: 2rem;
        transition: border 0.3s ease; /* Smooth transition for hover */
    }

    /* Sidebar container */
    [data-testid="stSidebar"] > div:first-child {
        background: rgba(15, 12, 41, 0.5); /* Matching sidebar background */
        border-right: 1px solid rgba(138, 43, 226, 0.3);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        border-radius: 0 15px 15px 0; /* Rounded on the right side */
        transition: box-shadow 0.3s ease;
    }

    /* ------------------- Interactive Hover Effects ------------------- */
    .main .block-container:hover,
    [data-testid="stSidebar"] > div:first-child:hover {
        border: 1px solid rgba(3, 218, 198, 0.6); /* Highlight with neon blue on hover */
        box-shadow: 0 0 25px rgba(3, 218, 198, 0.3); /* Add a glow effect */
    }

    /* ------------------- Typography & Colors ------------------- */
    h1, h2, h3, h4, h5, h6 {
        color: #BB86FC; /* Main headings in soft purple */
        font-weight: 700;
    }

    /* Metric values for a distinct look */
    [data-testid="stMetricValue"] {
        color: #03DAC6; /* Neon teal for metric numbers */
        font-size: 2.2rem;
        font-weight: 700;
    }

    /* Metric labels */
    [data-testid="stMetricLabel"] {
        color: #A9A9A9; /* Lighter grey for metric labels */
    }
    
    /* General text and fetched data emphasis */
    strong, .stMarkdown p {
        color: #E0E0E0;
    }
    .stMarkdown strong {
        color: #03DAC6; /* Make important fetched values stand out */
    }

    /* ------------------- Widget Styling ------------------- */
    /* Style for expanders to match the glass theme */
    .stExpander {
        background: rgba(40, 40, 80, 0.3);
        border: 1px solid rgba(138, 43, 226, 0.2);
        border-radius: 10px;
        padding: 0.5rem;
    }
    .stExpander:hover {
        border: 1px solid rgba(3, 218, 198, 0.4);
    }

    /* Style for tabs to be more integrated */
    [data-testid="stTabs"] {
        border-bottom: 1px solid #BB86FC;
    }
    
    [data-testid="stTab"] {
        background-color: transparent;
        color: #A9A9A9;
    }

    [data-testid="stTab"][aria-selected="true"] {
        background-color: rgba(187, 134, 252, 0.2);
        color: #BB86FC;
        border-radius: 5px 5px 0 0;
    }

    /* Button styling */
    .stButton>button {
        background-color: #03DAC6;
        color: #000000;
        border: none;
        border-radius: 8px;
        transition: background-color 0.3s, transform 0.2s;
    }
    .stButton>button:hover {
        background-color: #12F7D6;
        transform: scale(1.05);
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.title("📈 Option Pricing Visualizer")

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

@st.cache_data(ttl=3600)
def get_stock_info(ticker_symbol):
    try:
        return yf.Ticker(ticker_symbol).info
    except Exception:
        return {}

@st.cache_data(ttl=3600)
def get_stock_history(ticker_symbol, period):
    try:
        return yf.Ticker(ticker_symbol).history(period=period)
    except Exception:
        return pd.DataFrame()

with st.sidebar.expander("📈 Underlying Stock Parameters", expanded=True):
    current_ticker = st.session_state.get('ticker_input', 'AAPL')
    ticker = st.text_input("Enter Stock Ticker", value=current_ticker).upper()
    st.session_state['ticker_input'] = ticker

    info = get_stock_info(ticker)
    company_name = info.get('longName', '').strip()
    
    if company_name:
        st.write(f"**Company:** **{company_name}**")
    else:
        st.warning(f"Company name not found for '{ticker}'.")

    spot_price, vol_est, rf_fetch = 100.0, 0.20, 0.03
    currency = "$"
    
    hist = get_stock_history(ticker, "5d")
    if not hist.empty:
        spot_price = hist["Close"].iloc[-1]
        currency = "₹" if ticker.endswith(".NS") else "$"
        spot_help_text = f"Live Price: {currency}{spot_price:.2f}"

        hist30 = get_stock_history(ticker, "30d")["Close"]
        if not hist30.empty:
            log_ret = np.log(hist30 / hist30.shift(1)).dropna()
            vol_est = np.std(log_ret) * np.sqrt(252)
            vol_help_text = f"30d Annualized Volatility: {vol_est:.2%}"
        else:
            vol_help_text = "Could not estimate 30d volatility."
    else:
        spot_help_text = f"No data for '{ticker}'. Using default."
        vol_help_text = "Cannot estimate volatility. Using default."

    S = st.number_input("Spot Price", value=float(spot_price), min_value=0.01, format="%.2f", help=spot_help_text)
    sigma = st.number_input("Volatility (σ)", min_value=0.01, max_value=2.0, value=round(vol_est, 2), step=0.01, help=vol_help_text)

    rf_ticker, rf_name = ("^NSITEN", "India 10Y Bond") if ticker.endswith(".NS") else ("^IRX", "US 13W T-Bill")
    rf_data = get_stock_history(rf_ticker, "1d")["Close"]
    if not rf_data.empty:
        rf_fetch = rf_data.iloc[-1] / 100
        rf_help_text = f"Live {rf_name} rate: {rf_fetch:.3%}"
    else:
        rf_help_text = f"Could not fetch {rf_name} rate."
    
    r = st.number_input("Risk-Free Rate (r)", min_value=0.0, max_value=0.2, value=float(rf_fetch), step=0.001, format="%.3f", help=rf_help_text)

    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

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
    else: # Monte Carlo
        num_sims = kwargs.get('num_simulations', 10000)
        price = monte_carlo_option_pricing(S, K, T, r, sigma, option_type, num_sims)
        delta, gamma, theta, vega, rho = mc_greeks(S, K, T, r, sigma, option_type, num_sims)
    return price, delta, gamma, theta, vega, rho

# ------------------- Main Calculation & Display Block -------------------
model_params = {}
if selected_model == "Binomial Option Pricing":
    model_params['N'] = N_binomial
elif selected_model == "Monte Carlo Simulation":
    model_params['num_simulations'] = num_simulations_mc

with st.spinner(f"Calculating with {selected_model} model..."):
    call_price, cd, cg, ct, cv, cr = get_option_value_and_greeks(selected_model, S, K, T, r, sigma, "call", **model_params)
    put_price, pd, pg, pt, pv, pr = get_option_value_and_greeks(selected_model, S, K, T, r, sigma, "put", **model_params)

# --- Define Plotly template for dark theme ---
plotly_template = "plotly_dark"

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Summary", "💸 Payoff Diagram", "📊 Model Comparison", "📈 3D Surface", "🔥 Heatmaps", "🎯 Cross-Section"
])

with tab0:
    st.header(f"Option Valuation ({selected_model})")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Call Option")
        st.metric(label="Price", value=f"{currency} {call_price:.2f}")
        gcol1, gcol2 = st.columns(2)
        gcol1.metric(label="Delta (Δ)", value=f"{cd:.4f}")
        gcol2.metric(label="Gamma (Γ)", value=f"{cg:.4f}")
        gcol1.metric(label="Vega", value=f"{cv:.4f}")
        gcol2.metric(label="Theta (Θ)", value=f"{ct:.4f}")
        gcol1.metric(label="Rho (Ρ)", value=f"{cr:.4f}")

    with col2:
        st.subheader("Put Option")
        st.metric(label="Price", value=f"{currency} {put_price:.2f}")
        gcol1, gcol2 = st.columns(2)
        gcol1.metric(label="Delta (Δ)", value=f"{pd:.4f}")
        gcol2.metric(label="Gamma (Γ)", value=f"{pg:.4f}")
        gcol1.metric(label="Vega", value=f"{pv:.4f}")
        gcol2.metric(label="Theta (Θ)", valuef"{pt:.4f}")
        gcol1.metric(label="Rho (Ρ)", value=f"{pr:.4f}")

with tab1:
    st.header("Profit/Loss at Expiration")
    spot_range = np.linspace(S * 0.7, S * 1.3, 100)
    call_payoff = np.maximum(spot_range - K, 0) - call_price
    put_payoff = np.maximum(K - spot_range, 0) - put_price
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spot_range, y=call_payoff, mode='lines', name='Call Option P/L', line=dict(color='#03DAC6')))
    fig.add_trace(go.Scatter(x=spot_range, y=put_payoff, mode='lines', name='Put Option P/L', line=dict(color='#BB86FC')))
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    fig.add_vline(x=K, line_dash="dash", line_color="red", annotation_text="Strike Price")
    fig.update_layout(
        title="Option Payoff Profile",
        xaxis_title="Stock Price at Expiration",
        yaxis_title="Profit / Loss per Share",
        template=plotly_template,
        legend_title="Option Type",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Model Price Comparison")
    with st.spinner("Running all models for comparison..."):
        bs_call, bs_cd, bs_cg, bs_ct, bs_cv, bs_cr = get_option_value_and_greeks("Black-Scholes", S, K, T, r, sigma, "call")
        bs_put, bs_pd, bs_pg, bs_pt, bs_pv, bs_pr = get_option_value_and_greeks("Black-Scholes", S, K, T, r, sigma, "put")
        
        n_comp = 100 if selected_model != "Binomial Option Pricing" else N_binomial
        bi_call, bi_cd, bi_cg, bi_ct, bi_cv, bi_cr = get_option_value_and_greeks("Binomial Option Pricing", S, K, T, r, sigma, "call", N=n_comp)
        bi_put, bi_pd, bi_pg, bi_pt, bi_pv, bi_pr = get_option_value_and_greeks("Binomial Option Pricing", S, K, T, r, sigma, "put", N=n_comp)
        
        sims_comp = 10000 if selected_model != "Monte Carlo Simulation" else num_simulations_mc
        mc_call, mc_cd, mc_cg, mc_ct, mc_cv, mc_cr = get_option_value_and_greeks("Monte Carlo Simulation", S, K, T, r, sigma, "call", num_simulations=sims_comp)
        mc_put, mc_pd, mc_pg, mc_pt, mc_pv, mc_pr = get_option_value_and_greeks("Monte Carlo Simulation", S, K, T, r, sigma, "put", num_simulations=sims_comp)

    st.subheader("Call Option Comparison")
    st.dataframe({
        "Metric": ["Price", "Delta", "Gamma", "Theta", "Vega", "Rho"],
        "Black-Scholes": [bs_call, bs_cd, bs_cg, bs_ct, bs_cv, bs_cr],
        f"Binomial (N={n_comp})": [bi_call, bi_cd, bi_cg, bi_ct, bi_cv, bi_cr],
        f"Monte Carlo (Sims={sims_comp})": [mc_call, mc_cd, mc_cg, mc_ct, mc_cv, mc_cr],
    }, use_container_width=True)
    
    st.subheader("Put Option Comparison")
    st.dataframe({
        "Metric": ["Price", "Delta", "Gamma", "Theta", "Vega", "Rho"],
        "Black-Scholes": [bs_put, bs_pd, bs_pg, bs_pt, bs_pv, bs_pr],
        f"Binomial (N={n_comp})": [bi_put, bi_pd, bi_pg, bi_pt, bi_pv, bi_pr],
        f"Monte Carlo (Sims={sims_comp})": [mc_put, mc_pd, mc_pg, mc_pt, mc_pv, mc_pr],
    }, use_container_width=True)

with tab3:
    st.header(f"3D Price Surface ({selected_model})")
    def plot_3d(option_type, model, **kwargs):
        spot_range = np.linspace(0.5*S, 1.5*S, 30)
        time_range = np.linspace(T, 0.01, 30)
        Spot, Time = np.meshgrid(spot_range, time_range)
        Z = np.zeros_like(Spot)

        for i in range(Spot.shape[0]):
            for j in range(Spot.shape[1]):
                Z[i, j], _, _, _, _, _ = get_option_value_and_greeks(model, Spot[i, j], K, Time[i, j], r, sigma, option_type.lower(), **kwargs)

        fig = go.Figure(data=[go.Surface(x=Spot, y=Time, z=Z, colorscale='Viridis')])
        fig.update_layout(
            title=f"{option_type.capitalize()} Option Price vs. Spot and Time",
            scene=dict(xaxis_title="Spot Price", yaxis_title="Time to Maturity", zaxis_title="Option Price"),
            template=plotly_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, b=0, t=40))
        return fig

    st.plotly_chart(plot_3d("call", selected_model, **model_params), use_container_width=True)
    st.plotly_chart(plot_3d("put", selected_model, **model_params), use_container_width=True)

with tab4:
    st.header(f"Price Heatmaps vs. Spot & Volatility ({selected_model})")
    
    with st.expander("Adjust Heatmap Parameters"):
        min_spot = st.number_input("Min Spot Price", value=round(S * 0.8, 2))
        max_spot = st.number_input("Max Spot Price", value=round(S * 1.2, 2))
        min_vol = st.number_input("Min Volatility", value=max(0.01, round(sigma - 0.1, 2)), step=0.01)
        max_vol = st.number_input("Max Volatility", value=min(1.0, round(sigma + 0.1, 2)), step=0.01)
        num_points = st.slider("Heatmap Resolution", 5, 50, 10, 1, help="Grid resolution for the heatmap.")

    spot_range = np.linspace(min_spot, max_spot, num_points)
    vol_range = np.linspace(min_vol, max_vol, num_points)
    call_prices, put_prices = np.zeros((len(vol_range), len(spot_range))), np.zeros((len(vol_range), len(spot_range)))

    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            call_prices[i, j], _, _, _, _, _ = get_option_value_and_greeks(selected_model, spot, K, T, r, vol, "call", **model_params)
            put_prices[i, j], _, _, _, _, _ = get_option_value_and_greeks(selected_model, spot, K, T, r, vol, "put", **model_params)

    def plot_heatmap(prices, title):
        fig = go.Figure(data=go.Heatmap(z=prices, x=spot_range, y=vol_range, colorscale='Viridis', hoverongaps=False))
        fig.update_layout(
            title=title, xaxis_title="Spot Price", yaxis_title="Volatility",
            template=plotly_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_heatmap(call_prices, "Call Option Prices"), use_container_width=True)
    with col2:
        st.plotly_chart(plot_heatmap(put_prices, "Put Option Prices"), use_container_width=True)

with tab5:
    st.header("Sensitivity Analysis")
    col1, col2, col3 = st.columns(3)
    option_type_cs = col1.selectbox("Option Type", ["Call", "Put"], key="opt_type_cs")
    y_axis_value = col2.selectbox("Y-Axis Value", ["Price", "Delta", "Gamma", "Theta", "Vega", "Rho"], key="y_axis_cs")
    varying_param = col3.selectbox("Parameter to Vary", ["Spot Price", "Strike Price", "Volatility", "Time to Maturity"], key="var_param_cs")
    
    fixed = {"S": S, "K": K, "T": T, "r": r, "sigma": sigma}
    param_map = {"Spot Price": "S", "Strike Price": "K", "Volatility": "sigma", "Time to Maturity": "T"}
    var_param_key = param_map[varying_param]

    x_vals = np.linspace(0.7 * fixed[var_param_key], 1.3 * fixed[var_param_key], 100)
    y_vals = []
    
    with st.spinner("Generating sensitivity graph..."):
        for val in x_vals:
            temp = fixed.copy()
            temp[var_param_key] = val
            price, delta, gamma, theta, vega, rho = get_option_value_and_greeks(selected_model, temp["S"], temp["K"], temp["T"], temp["r"], temp["sigma"], option_type_cs.lower(), **model_params)
            greeks_map = {"Price": price, "Delta": delta, "Gamma": gamma, "Theta": theta, "Vega": vega, "Rho": rho}
            y_vals.append(greeks_map[y_axis_value])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', line=dict(color='#03DAC6')))
    fig.update_layout(
        title=f"{option_type_cs} {y_axis_value} vs. {varying_param} ({selected_model})",
        xaxis_title=varying_param, yaxis_title=y_axis_value,
        template=plotly_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
