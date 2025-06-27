import streamlit as st
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from scipy.stats import norm
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Modern Option Pricer",
    page_icon="📈"
)

# --- Main Title and Subheader ---
st.title("📈 Modern Option Pricing Visualizer")
st.markdown("A sleek and intuitive dashboard for pricing options and analyzing their sensitivities using various financial models.")

# --- Custom CSS for a Modern Dark Theme ---
st.markdown(
    """
    <style>
    /* A modern, clean color scheme for dark mode */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

    body {
        font-family: 'Roboto', sans-serif;
    }

    /* Primary Accent Color for Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #6cb2eb; /* A pleasing, modern blue */
    }

    /* High-contrast color for primary data points (metric values, fetched prices) */
    [data-testid="stMetricValue"], strong {
        color: #f0f0f0; /* Off-white for clarity */
    }

    /* Softer color for metric labels to de-emphasize them slightly */
    [data-testid="stMetricLabel"] {
        color: #a9a9a9; /* Light grey */
    }

    /* Styling for buttons to match the theme */
    .stButton>button {
        color: #f0f0f0;
        background-color: #6cb2eb;
        border: 2px solid #6cb2eb;
        border-radius: 8px;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #f0f0f0;
        color: #6cb2eb;
        border: 2px solid #f0f0f0;
    }
    .stButton>button:focus {
        box-shadow: none !important;
        outline: none !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

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
    
    # Use a shared set of random numbers for better stability in finite differences
    random_numbers = np.random.standard_normal(num_simulations)

    def mc_price(s, k, t, r_rate, vol, sims):
        st_prices = s * np.exp((r_rate - 0.5 * vol**2) * t + vol * np.sqrt(t) * random_numbers)
        if option_type.lower() == "call":
            payoffs = np.maximum(0, st_prices - k)
        else:
            payoffs = np.maximum(0, k - st_prices)
        return np.exp(-r_rate * t) * np.mean(payoffs)

    price_mid = mc_price(S, K, T, r, sigma, num_simulations)
    
    price_S_up = mc_price(S + dS, K, T, r, sigma, num_simulations)
    price_S_down = mc_price(S - dS, K, T, r, sigma, num_simulations)
    delta = (price_S_up - price_S_down) / (2 * dS)
    gamma = (price_S_up - 2 * price_mid + price_S_down) / (dS ** 2)

    price_vol_up = mc_price(S, K, T, r, sigma + d_sigma, num_simulations)
    vega = (price_vol_up - price_mid) / d_sigma

    price_t_down = mc_price(S, K, T - dT, r, sigma, num_simulations)
    theta = (price_t_down - price_mid) / dT

    price_r_up = mc_price(S, K, T, r + d_r, sigma, num_simulations)
    rho = (price_r_up - price_mid) / d_r

    return delta, gamma, theta, vega, rho

# ------------------- Sidebar Controls -------------------
st.sidebar.header("🛠️ Model Configuration")
selected_model = st.sidebar.selectbox("Select Pricing Model", ["Black-Scholes", "Binomial Option Pricing", "Monte Carlo Simulation"])

if selected_model == "Binomial Option Pricing":
    N_binomial = st.sidebar.slider("Number of Steps (N)", min_value=10, max_value=1000, value=100, step=10, help="Higher N increases accuracy but slows down calculation.")
elif selected_model == "Monte Carlo Simulation":
    num_simulations_mc = st.sidebar.slider("Number of Simulations", min_value=1000, max_value=100000, value=10000, step=1000, help="More simulations lead to more accurate, stable results.")

# --- Data Fetching Functions with Caching ---
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

# --- Sidebar Parameter Inputs ---
with st.sidebar.expander("📈 Underlying Asset Parameters", expanded=True):
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, GOOGL, MSFT.NS)", value=st.session_state.get('ticker_input', 'AAPL')).upper()
    st.session_state['ticker_input'] = ticker

    # Fetch and display stock info
    info = get_stock_info(ticker)
    company_name = info.get('longName', '').strip()
    if company_name:
        st.write(f"**Company:** {company_name}")
    else:
        st.warning(f"Could not find data for '{ticker}'. Using default values.")

    # Fetch live data and set defaults
    spot_price_default, vol_default, r_default = 100.0, 0.20, 0.03
    currency = "$"
    hist_5d = get_stock_history(ticker, "5d")
    
    if not hist_5d.empty:
        spot_price_default = hist_5d["Close"].iloc[-1]
        currency = "₹" if ticker.endswith(".NS") else "$"
        
        hist_30d = get_stock_history(ticker, "30d")["Close"]
        if not hist_30d.empty:
            log_ret = np.log(hist_30d / hist_30d.shift(1)).dropna()
            vol_default = np.std(log_ret) * np.sqrt(252)
    
    S = st.number_input("Spot Price", value=float(spot_price_default), min_value=0.01, format="%.2f", help=f"Current price of the underlying asset. Last fetched for {ticker}: {currency}{spot_price_default:.2f}")
    sigma = st.number_input("Volatility (σ)", value=round(vol_default, 3), min_value=0.01, max_value=2.0, step=0.001, format="%.3f", help=f"Annualized volatility. 30-day estimate for {ticker}: {vol_default:.3f}")

    # Dynamic Risk-Free Rate
    rf_ticker = "^NSITEN" if ticker.endswith(".NS") else "^IRX"
    rf_name = "India 10Y Bond" if ticker.endswith(".NS") else "US 13W T-Bill"
    rf_data = get_stock_history(rf_ticker, "1d")
    if not rf_data.empty:
        r_default = rf_data["Close"].iloc[-1] / 100
        
    r = st.number_input("Risk-Free Rate (r)", value=float(r_default), min_value=0.0, max_value=0.2, step=0.001, format="%.3f", help=f"Risk-free interest rate. Fetched from {rf_name}: {r_default:.3%}")

with st.sidebar.expander("⚙️ Option Contract Parameters", expanded=True):
    K = st.number_input("Strike Price", value=float(spot_price_default), min_value=0.01, format="%.2f")
    T = st.number_input("Time to Maturity (years)", value=0.5, min_value=0.01, max_value=5.0, step=0.01)

# ------------------- Main Calculation Block -------------------
def get_option_value_and_greeks(model, S, K, T, r, sigma, option_type, **kwargs):
    model_map = {
        "Black-Scholes": (black_scholes, bs_greeks),
        "Binomial Option Pricing": (binomial_option_pricing, binomial_greeks),
        "Monte Carlo Simulation": (monte_carlo_option_pricing, mc_greeks)
    }
    price_func, greeks_func = model_map[model]
    
    price = price_func(S, K, T, r, sigma, option_type, **kwargs)
    delta, gamma, theta, vega, rho = greeks_func(S, K, T, r, sigma, option_type, **kwargs)
    
    return price, delta, gamma, theta, vega, rho

model_params = {}
if selected_model == "Binomial Option Pricing":
    model_params['N'] = N_binomial
elif selected_model == "Monte Carlo Simulation":
    model_params['num_simulations'] = num_simulations_mc

with st.spinner(f"Calculating with {selected_model} model..."):
    call_price, cd, cg, ct, cv, cr = get_option_value_and_greeks(selected_model, S, K, T, r, sigma, "call", **model_params)
    put_price, pd, pg, pt, pv, pr = get_option_value_and_greeks(selected_model, S, K, T, r, sigma, "put", **model_params)

# ------------------- TABS for Displaying Results -------------------
tab_summary, tab_payoff, tab_compare, tab_3d, tab_heatmap, tab_sensitivity = st.tabs([
    "📊 Summary", "📈 Payoff Diagram", "🆚 Model Comparison", "🧊 3D Surface", "🌡️ Heatmaps", "🔪 Sensitivity"
])

# --- Tab 1: Summary ---
with tab_summary:
    st.header(f"Valuation Results ({selected_model})")
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Call Option")
            st.metric(label="Price", value=f"{currency} {call_price:.3f}")
            gcol1, gcol2 = st.columns(2)
            gcol1.metric("Delta (Δ)", f"{cd:.4f}")
            gcol2.metric("Gamma (Γ)", f"{cg:.4f}")
            gcol1.metric("Vega", f"{cv:.4f}")
            gcol2.metric("Theta (Θ)", f"{ct:.4f}")
            gcol1.metric("Rho (Ρ)", f"{cr:.4f}")

        with col2:
            st.subheader("Put Option")
            st.metric(label="Price", value=f"{currency} {put_price:.3f}")
            gcol1, gcol2 = st.columns(2)
            gcol1.metric("Delta (Δ)", f"{pd:.4f}")
            gcol2.metric("Gamma (Γ)", f"{pg:.4f}")
            gcol1.metric("Vega", f"{pv:.4f}")
            gcol2.metric("Theta (Θ)", f"{pt:.4f}")
            gcol1.metric("Rho (Ρ)", f"{pr:.4f}")

# --- Tab 2: Payoff Diagram ---
with tab_payoff:
    st.header("Profit/Loss Profile at Expiration")
    spot_range = np.linspace(S * 0.75, S * 1.25, 100)
    call_pnl = np.maximum(spot_range - K, 0) - call_price
    put_pnl = np.maximum(K - spot_range, 0) - put_price
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spot_range, y=call_pnl, mode='lines', name='Call Option P/L', line=dict(color='#6cb2eb', width=3)))
    fig.add_trace(go.Scatter(x=spot_range, y=put_pnl, mode='lines', name='Put Option P/L', line=dict(color='#ff6b6b', width=3)))
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    fig.add_vline(x=K, line_dash="dot", line_color="white", annotation_text="Strike", annotation_position="top left")

    fig.update_layout(
        title="Option Payoff at Expiration",
        xaxis_title="Stock Price at Expiration",
        yaxis_title="Profit / Loss per Share",
        legend_title="Option Type",
        template='plotly_dark'
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 3: Model Comparison ---
with tab_compare:
    st.header("Model Price & Greeks Comparison")
    with st.spinner("Running all models for comparison..."):
        # Black-Scholes
        bs_c, bs_cd, bs_cg, bs_ct, bs_cv, bs_cr = get_option_value_and_greeks("Black-Scholes", S, K, T, r, sigma, "call")
        bs_p, bs_pd, bs_pg, bs_pt, bs_pv, bs_pr = get_option_value_and_greeks("Black-Scholes", S, K, T, r, sigma, "put")
        # Binomial
        bi_c, bi_cd, bi_cg, bi_ct, bi_cv, bi_cr = get_option_value_and_greeks("Binomial Option Pricing", S, K, T, r, sigma, "call", N=100)
        bi_p, bi_pd, bi_pg, bi_pt, bi_pv, bi_pr = get_option_value_and_greeks("Binomial Option Pricing", S, K, T, r, sigma, "put", N=100)
        # Monte Carlo
        mc_c, mc_cd, mc_cg, mc_ct, mc_cv, mc_cr = get_option_value_and_greeks("Monte Carlo Simulation", S, K, T, r, sigma, "call", num_simulations=10000)
        mc_p, mc_pd, mc_pg, mc_pt, mc_pv, mc_pr = get_option_value_and_greeks("Monte Carlo Simulation", S, K, T, r, sigma, "put", num_simulations=10000)

    with st.container(border=True):
        st.subheader("Call Option Comparison")
        df_call = pd.DataFrame({
            "Metric": ["Price", "Delta", "Gamma", "Theta", "Vega", "Rho"],
            "Black-Scholes": [bs_c, bs_cd, bs_cg, bs_ct, bs_cv, bs_cr],
            "Binomial (N=100)": [bi_c, bi_cd, bi_cg, bi_ct, bi_cv, bi_cr],
            "Monte Carlo (Sims=10k)": [mc_c, mc_cd, mc_cg, mc_ct, mc_cv, mc_cr],
        }).set_index("Metric")
        st.dataframe(df_call.style.format("{:.4f}"), use_container_width=True)
    
    st.divider()

    with st.container(border=True):
        st.subheader("Put Option Comparison")
        df_put = pd.DataFrame({
            "Metric": ["Price", "Delta", "Gamma", "Theta", "Vega", "Rho"],
            "Black-Scholes": [bs_p, bs_pd, bs_pg, bs_pt, bs_pv, bs_pr],
            "Binomial (N=100)": [bi_p, bi_pd, bi_pg, bi_pt, bi_pv, bi_pr],
            "Monte Carlo (Sims=10k)": [mc_p, mc_pd, mc_pg, mc_pt, mc_pv, mc_pr],
        }).set_index("Metric")
        st.dataframe(df_put.style.format("{:.4f}"), use_container_width=True)

# --- Tab 4: 3D Surface ---
with tab_3d:
    st.header(f"3D Price Surface ({selected_model})")
    
    @st.cache_data
    def calculate_3d_surface(option_type, model, S, K, T, r, sigma, **kwargs):
        spot_range = np.linspace(0.5 * S, 1.5 * S, 30)
        time_range = np.linspace(T, 0.01, 30)
        Spot, Time = np.meshgrid(spot_range, time_range)
        Z = np.array([
            get_option_value_and_greeks(model, s, K, t, r, sigma, option_type.lower(), **kwargs)[0]
            for s, t in zip(np.ravel(Spot), np.ravel(Time))
        ]).reshape(Spot.shape)
        return Spot, Time, Z

    def plot_3d(Spot, Time, Z, option_type):
        fig = go.Figure(data=[go.Surface(x=Spot, y=Time, z=Z, colorscale='Blues', showscale=False)])
        fig.update_layout(
            title=f"{option_type.capitalize()} Price vs. Spot & Time",
            scene=dict(xaxis_title="Spot Price", yaxis_title="Time to Maturity (Yrs)", zaxis_title="Option Price"),
            margin=dict(l=0, r=0, b=0, t=40),
            template='plotly_dark'
        )
        return fig

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Call Option Surface")
        Spot, Time, Z_call = calculate_3d_surface("call", selected_model, S, K, T, r, sigma, **model_params)
        st.plotly_chart(plot_3d(Spot, Time, Z_call, "call"), use_container_width=True)
    with col2:
        st.subheader("Put Option Surface")
        Spot, Time, Z_put = calculate_3d_surface("put", selected_model, S, K, T, r, sigma, **model_params)
        st.plotly_chart(plot_3d(Spot, Time, Z_put, "put"), use_container_width=True)

# --- Tab 5: Heatmaps ---
with tab_heatmap:
    st.header(f"Price Heatmap vs. Spot & Volatility ({selected_model})")
    
    with st.expander("Adjust Heatmap Parameters"):
        col1, col2 = st.columns(2)
        min_spot = col1.number_input("Min Spot", value=round(S * 0.8, 2))
        max_spot = col1.number_input("Max Spot", value=round(S * 1.2, 2))
        min_vol = col2.number_input("Min Vol", value=max(0.01, round(sigma - 0.1, 2)), format="%.2f")
        max_vol = col2.number_input("Max Vol", value=min(1.0, round(sigma + 0.1, 2)), format="%.2f")
    
    resolution = st.slider("Heatmap Resolution", 5, 20, 10, help="Grid size for the heatmap. Higher is more detailed.")
    
    @st.cache_data
    def calculate_heatmap_data(_model, _min_spot, _max_spot, _min_vol, _max_vol, _resolution, _K, _T, _r, **kwargs):
        spot_range = np.linspace(_min_spot, _max_spot, _resolution)
        vol_range = np.linspace(_min_vol, _max_vol, _resolution)
        call_prices = np.zeros((_resolution, _resolution))
        put_prices = np.zeros((_resolution, _resolution))

        for i, vol in enumerate(vol_range):
            for j, spot in enumerate(spot_range):
                call_prices[i, j] = get_option_value_and_greeks(_model, spot, _K, _T, _r, vol, "call", **kwargs)[0]
                put_prices[i, j] = get_option_value_and_greeks(_model, spot, _K, _T, _r, vol, "put", **kwargs)[0]
        return spot_range, vol_range, call_prices, put_prices

    spot_range, vol_range, call_prices, put_prices = calculate_heatmap_data(
        selected_model, min_spot, max_spot, min_vol, max_vol, resolution, K, T, r, **model_params
    )

    def plot_heatmap(prices, x_vals, y_vals, title):
        fig = go.Figure(data=go.Heatmap(z=prices, x=x_vals, y=y_vals, colorscale='Blues', hoverongaps=False))
        fig.update_layout(title=title, xaxis_title="Spot Price", yaxis_title="Volatility", template='plotly_dark')
        return fig

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_heatmap(call_prices, spot_range, vol_range, "Call Option Prices"), use_container_width=True)
    with col2:
        st.plotly_chart(plot_heatmap(put_prices, spot_range, vol_range, "Put Option Prices"), use_container_width=True)

# --- Tab 6: Sensitivity Analysis ---
with tab_sensitivity:
    st.header("Greeks Sensitivity Analysis")
    
    col1, col2, col3 = st.columns(3)
    option_type_cs = col1.selectbox("Option Type", ["Call", "Put"], key="cs_opt_type")
    y_axis_cs = col2.selectbox("Y-Axis (Greek)", ["Price", "Delta", "Gamma", "Theta", "Vega", "Rho"], key="cs_y_axis")
    varying_param_cs = col3.selectbox("X-Axis (Varying Param)", ["Spot Price", "Strike Price", "Volatility", "Time to Maturity"], key="cs_x_axis")
    
    param_map = {"Spot Price": "S", "Strike Price": "K", "Volatility": "sigma", "Time to Maturity": "T"}
    var_key = param_map[varying_param_cs]
    
    @st.cache_data
    def calculate_sensitivity_data(_model, _option_type, _y_axis, _var_key, S, K, T, r, sigma, **kwargs):
        fixed_params = {"S": S, "K": K, "T": T, "r": r, "sigma": sigma}
        x_vals = np.linspace(0.7 * fixed_params[_var_key], 1.3 * fixed_params[_var_key], 100)
        y_vals = []
        greeks_map = {"Price": 0, "Delta": 1, "Gamma": 2, "Theta": 3, "Vega": 4, "Rho": 5}
        
        for val in x_vals:
            temp_params = fixed_params.copy()
            temp_params[_var_key] = val
            results = get_option_value_and_greeks(_model, temp_params["S"], temp_params["K"], temp_params["T"], temp_params["r"], temp_params["sigma"], _option_type.lower(), **kwargs)
            y_vals.append(results[greeks_map[_y_axis]])
        return x_vals, y_vals

    x_vals, y_vals = calculate_sensitivity_data(selected_model, option_type_cs, y_axis_cs, var_key, S, K, T, r, sigma, **model_params)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', line=dict(color='#6cb2eb', width=2)))
    fig.update_layout(
        title=f"{option_type_cs} {y_axis_cs} vs. {varying_param_cs}",
        xaxis_title=varying_param_cs,
        yaxis_title=y_axis_cs,
        template='plotly_dark'
    )
    st.plotly_chart(fig, use_container_width=True)
