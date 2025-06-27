import streamlit as st
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from scipy.stats import norm
import pandas as pd

st.set_page_config(layout="wide", page_title="Option Pricing Visualizer")
st.title("📈 Option Pricing Visualizer")

# ------------------- Data for Stock Markets (Modified - Removed predefined stocks) -------------------
MARKETS = {
    "🇺🇸 US Exchanges": {
        "currency": "$",
        "rf_ticker": "^IRX",  # 13 Week Treasury Bill
        "rf_name": "US 13W T-Bill",
    },
    "🇮🇳 Indian Exchanges": {
        "currency": "₹",
        "rf_ticker": "^NSITEN", # India 10Y Bond Yield (keeping as per original, but handling errors)
        "rf_name": "India 10Y Bond",
    },
    "🇪🇺 European Exchanges": {
        "currency": "€",
        "rf_ticker": "DE10Y.SG", # German 10-Year Bond as a proxy for Eurozone rate
        "rf_name": "German 10Y Bund",
    }
}

# ------------------- Utility Functions -------------------
@st.cache_data
def get_risk_free_rate(ticker):
    try:
        data = yf.download(ticker, period="5d", interval="1d")
        if not data.empty:
            rate = data['Close'].iloc[-1]
            # Heuristic to check if rate is likely a percentage (e.g., 5.0 for 5%) and convert to decimal
            if rate > 1.0 and rate > 20: # If rate is like 294.78 or 5.0 (for 5%), convert
                return rate / 100.0
            return rate
        else:
            st.warning(f"No data found for risk-free rate ticker: {ticker}. Using default of 5%.")
            return 0.05
    except Exception as e:
        st.error(f"Error fetching risk-free rate for {ticker}: {e}. Using default of 5%.")
        return 0.05

@st.cache_data
def get_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        current_price = info.get('regularMarketPrice')
        company_name = info.get('longName')
        if current_price is None or company_name is None:
            st.error(f"Could not retrieve full information for ticker: {ticker}. Please check the ticker symbol.")
            return None, None
        return current_price, company_name
    except Exception as e:
        st.error(f"Error fetching stock data for {ticker}: {e}. Please ensure it's a valid ticker.")
        return None, None

# ------------------- Black-Scholes Model -------------------
def black_scholes(S, K, T, r, sigma, option_type="call"):
    """Calculates the Black-Scholes option price."""
    if T <= 0 or sigma <= 0:
        if option_type == 'call':
            return max(0.0, S - K)
        else:
            return max(0.0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type.lower() == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # Put
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_greeks(S, K, T, r, sigma, option_type="call"):
    """Calculates the Greeks for the Black-Scholes model."""
    if T <= 0 or sigma <= 0:
        delta = 1.0 if S > K else (-1.0 if S < K else 0.0)
        if option_type != 'call': delta -= 1.0
        return (delta, 0.0, 0.0, 0.0, 0.0)

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
    if T <= 0:
        return max(0.0, S - K) if option_type == 'call' else max(0.0, K - S)
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    if not (0 < p < 1):
        st.warning(f"Binomial model instability detected (p={p:.4f}). Check parameters. The model requires 0 < p < 1 for no-arbitrage.")
        p = max(0, min(1, p)) # Clamp p to prevent errors, though results are unreliable

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
    if T <= 0 or sigma <= 0:
        delta = 1.0 if S > K else (-1.0 if S < K else 0.0)
        if option_type != 'call': delta -= 1.0
        return (delta, 0.0, 0.0, 0.0, 0.0)

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
    vega = (price_vol_up - price_vol_down) / (2*d_sigma)
    price_t_down = binomial_option_pricing(S, K, T - dT, r, sigma, option_type, N)
    theta = (price_t_down - price_mid) / dT
    price_r_up = binomial_option_pricing(S, K, T, r + d_r, sigma, option_type, N)
    price_r_down = binomial_option_pricing(S, K, T, r - d_r, sigma, option_type, N)
    rho = (price_r_up - price_r_down) / (2 * d_r)

    return delta, gamma, theta, vega, rho

# ------------------- Monte Carlo Simulation Model -------------------
def monte_carlo_option_pricing(S, K, T, r, sigma, option_type="call", num_simulations=10000):
    """Calculates option price using Monte Carlo simulation."""
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if option_type == 'call' else max(0.0, K - S)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * np.random.standard_normal(num_simulations))

    if option_type.lower() == "call":
        payoffs = np.maximum(0, ST - K)
    else: # Put
        payoffs = np.maximum(0, K - ST)

    return np.exp(-r * T) * np.mean(payoffs)

def mc_greeks(S, K, T, r, sigma, option_type="call", num_simulations=10000):
    """Calculates Greeks using Monte Carlo (finite differences)."""
    if T <= 0 or sigma <= 0:
        delta = 1.0 if S > K else (-1.0 if S < K else 0.0)
        if option_type != 'call': delta -= 1.0
        return (delta, 0.0, 0.0, 0.0, 0.0)

    dS = S * 0.01
    dT = max(0.001, T / 100) # Ensure dT is not zero
    d_sigma = sigma * 0.01
    d_r = r * 0.01 if r > 0 else 0.0001

    price_mid = monte_carlo_option_pricing(S, K, T, r, sigma, option_type, num_simulations)
    price_S_up = monte_carlo_option_pricing(S + dS, K, T, r, sigma, option_type, num_simulations)
    price_S_down = monte_carlo_option_pricing(S - dS, K, T, r, sigma, option_type, num_simulations)
    delta = (price_S_up - price_S_down) / (2 * dS)
    gamma = (price_S_up - 2 * price_mid + price_S_down) / (dS ** 2)
    price_vol_up = monte_carlo_option_pricing(S, K, T, r, sigma + d_sigma, option_type, num_simulations)
    price_vol_down = monte_carlo_option_pricing(S, K, T, r, sigma - d_sigma, option_type, num_simulations)
    vega = (price_vol_up - price_vol_down) / (2*d_sigma)
    price_t_down = monte_carlo_option_pricing(S, K, T - dT, r, sigma, option_type, num_simulations)
    theta = (price_t_down - price_mid) / dT
    price_r_up = monte_carlo_option_pricing(S, K, T, r + d_r, sigma, option_type, num_simulations)
    price_r_down = monte_carlo_option_pricing(S, K, T, r - d_r, sigma, option_type, num_simulations)
    rho = (price_r_up - price_r_down) / (2 * d_r)

    return delta, gamma, theta, vega, rho

# ------------------- Main App Logic -------------------
def get_option_value_and_greeks(model, S, K, T, r, sigma, option_type, **model_params):
    if model == "Black-Scholes":
        price = black_scholes(S, K, T, r, sigma, option_type)
        delta, gamma, theta, vega, rho = bs_greeks(S, K, T, r, sigma, option_type)
    elif model == "Binomial Option Pricing":
        N = model_params.get("N", 100)
        price = binomial_option_pricing(S, K, T, r, sigma, option_type, N)
        delta, gamma, theta, vega, rho = binomial_greeks(S, K, T, r, sigma, option_type, N)
    elif model == "Monte Carlo Simulation":
        num_simulations = model_params.get("num_simulations", 10000)
        price = monte_carlo_option_pricing(S, K, T, r, sigma, option_type, num_simulations)
        delta, gamma, theta, vega, rho = mc_greeks(S, K, T, r, sigma, option_type, num_simulations)
    return price, delta, gamma, theta, vega, rho

st.sidebar.markdown("## 🔧 Configure Parameters")
selected_model = st.sidebar.selectbox("Select Pricing Model", ["Black-Scholes", "Binomial Option Pricing", "Monte Carlo Simulation"])

with st.sidebar.expander("🌍 Market Data", expanded=True):
    selected_market = st.selectbox("Select Market", options=list(MARKETS.keys()))
    current_market = MARKETS[selected_market]
    currency = current_market["currency"]
    rf_ticker = current_market["rf_ticker"]
    rf_name = current_market["rf_name"]

    st.markdown(f"**Risk-Free Rate Ticker:** {rf_ticker} ({rf_name})")
    
    # Live Stock Ticker Input
    stock_ticker_input = st.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL").upper()
    
    company_name_display = "No stock selected"
    spot_price = None

    if stock_ticker_input:
        fetched_spot_price, fetched_company_name = get_stock_info(stock_ticker_input)
        if fetched_spot_price is not None and fetched_company_name is not None:
            spot_price = fetched_spot_price
            company_name_display = fetched_company_name
        else:
            st.warning(f"Could not find valid data for ticker: {stock_ticker_input}. Using a default spot price of 100.")
            spot_price = 100.0 # Fallback default
    else:
        spot_price = 100.0 # Default if nothing is entered

    st.info(f"Company: {company_name_display}") # Information icon display

    r_fetched = get_risk_free_rate(rf_ticker)
    
with st.sidebar.expander("⚙️ Option Parameters", expanded=True):
    K = st.number_input(f"Strike Price ({currency})", value=100.0, min_value=0.01)
    T = st.number_input("Time to Maturity (Years)", value=1.0, min_value=0.001, format="%.3f")
    sigma = st.number_input("Volatility (σ)", min_value=0.01, max_value=5.0, value=0.20, step=0.01) # Increased max_value
    r = st.number_input("Risk-Free Rate (r)", min_value=0.0, max_value=5.0, value=r_fetched, step=0.001) # Increased max_value
    option_type_cs = st.radio("Option Type", ["Call", "Put"])

# Ensure Spot Price (S) is obtained after the ticker input
S = st.sidebar.number_input(f"Spot Price ({currency})", value=float(spot_price), min_value=0.01)


with st.container():
    st.markdown("---")
    st.header("Option Pricing and Greeks")

    # Pass parameters to function
    price, delta, gamma, theta, vega, rho = get_option_value_and_greeks(
        selected_model, S, K, T, r, sigma, option_type_cs.lower(),
        N=st.session_state.get("binomial_n", 100),
        num_simulations=st.session_state.get("mc_sims", 10000)
    )

    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    with col1:
        st.metric(label="Option Price", value=f"{currency}{price:.4f}")
    with col2:
        st.metric(label="Delta (Δ)", value=f"{delta:.4f}")
    with col3:
        st.metric(label="Gamma (Γ)", value=f"{gamma:.4f}")
    with col4:
        st.metric(label="Theta (Θ)", value=f"{theta:.4f}")
    with col5:
        st.metric(label="Vega (ν)", value=f"{vega:.4f}")
    with col6:
        st.metric(label="Rho (Ρ)", value=f"{rho:.4f}")

    st.markdown("---")
    st.header("Sensitivity Analysis")

    varying_param = st.selectbox(
        "Varying Parameter",
        ["Spot Price", "Strike Price", "Volatility", "Time to Maturity"]
    )
    y_axis_value = st.selectbox(
        "Y-Axis Value",
        ["Price", "Delta", "Gamma", "Theta", "Vega", "Rho"]
    )

    fixed = {"S": S, "K": K, "T": T, "r": r, "sigma": sigma}
    param_map = {"Spot Price": "S", "Strike Price": "K", "Volatility": "sigma", "Time to Maturity": "T"}
    var_param_key = param_map[varying_param]

    x_vals = np.linspace(0.7 * fixed[var_param_key], 1.3 * fixed[var_param_key], 100)
    y_vals = []

    with st.spinner(f"Generating sensitivity graph for {varying_param} vs {y_axis_value}..."):
        for val in x_vals:
            temp = fixed.copy()
            temp[var_param_key] = val
            if var_param_key == 'T' and val <= 0: continue # Avoid error on x-axis
            if var_param_key == 'sigma' and val <= 0: continue
            
            # Recalculate based on selected model
            price, delta, gamma, theta, vega, rho = get_option_value_and_greeks(selected_model, temp["S"], temp["K"], temp["T"], temp["r"], temp["sigma"], option_type_cs.lower(),
                N=st.session_state.get("binomial_n", 100),
                num_simulations=st.session_state.get("mc_sims", 10000))

            greeks_map = {"Price": price, "Delta": delta, "Gamma": gamma, "Theta": theta, "Vega": vega, "Rho": rho}
            y_vals.append(greeks_map[y_axis_value])
            
    fig = go.Figure()
    valid_x_vals = [x for x, y in zip(x_vals, y_vals) if x > 0 and not np.isnan(y)]
    valid_y_vals = [y for x, y in zip(x_vals, y_vals) if x > 0 and not np.isnan(y)]

    fig.add_trace(go.Scatter(x=valid_x_vals, y=valid_y_vals, mode='lines', name=y_axis_value))
    fig.update_layout(
        title=f"{y_axis_value} vs {varying_param}",
        xaxis_title=varying_param,
        yaxis_title=y_axis_value
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.header("Model Parameters & Comparison")

    col_model_params, col_comparison = st.columns([1, 2])

    with col_model_params:
        st.subheader("Model Specific Parameters")
        if selected_model == "Binomial Option Pricing":
            n_comp = st.slider("Number of Steps (N)", 10, 500, 100, key="binomial_n")
        elif selected_model == "Monte Carlo Simulation":
            sims_comp = st.slider("Number of Simulations", 1000, 100000, 10000, key="mc_sims")
        else:
            st.write("No specific parameters for Black-Scholes.")

    with col_comparison:
        st.header("Model Price Comparison")
        # Logic for model comparison remains mostly the same
        with st.spinner("Running all models for comparison..."):
            # Black-Scholes values
            bs_call, bs_cd, bs_cg, bs_ct, bs_v, bs_r = get_option_value_and_greeks(
                "Black-Scholes", S, K, T, r, sigma, "call"
            )
            bs_put, bs_pd, bs_pg, bs_pt, bs_pv, bs_pr = get_option_value_and_greeks(
                "Black-Scholes", S, K, T, r, sigma, "put"
            )

            # Binomial values
            n_comp = st.session_state.get("binomial_n", 100) # Ensure N is available
            bi_call, bi_cd, bi_cg, bi_ct, bi_v, bi_r = get_option_value_and_greeks(
                "Binomial Option Pricing", S, K, T, r, sigma, "call", N=n_comp
            )
            bi_put, bi_pd, bi_pg, bi_pt, bi_pv, bi_pr = get_option_value_and_greeks(
                "Binomial Option Pricing", S, K, T, r, sigma, "put", N=n_comp
            )

            # Monte Carlo values
            sims_comp = st.session_state.get("mc_sims", 10000) # Ensure num_simulations is available
            mc_call, mc_cd, mc_cg, mc_ct, mc_v, mc_r = get_option_value_and_greeks(
                "Monte Carlo Simulation", S, K, T, r, sigma, "call", num_simulations=sims_comp
            )
            mc_put, mc_pd, mc_pg, mc_pt, mc_pv, mc_pr = get_option_value_and_greeks(
                "Monte Carlo Simulation", S, K, T, r, sigma, "put", num_simulations=sims_comp
            )

        metrics = ["Price", "Delta", "Gamma", "Theta", "Vega", "Rho"]

        # Call data
        call_df_data = {
            "Metric": metrics,
            f"Black-Scholes": [bs_call, bs_cd, bs_cg, bs_ct, bs_v, bs_r], # Fixed variable names
            f"Binomial (N={n_comp})": [bi_call, bi_cd, bi_cg, bi_ct, bi_v, bi_r], # Fixed variable names
            f"Monte Carlo (Sims={sims_comp})": [mc_call, mc_cd, mc_cg, mc_ct, mc_v, mc_r], # Fixed variable names
        }
        call_df = pd.DataFrame(call_df_data).set_index("Metric")

        # Put data
        put_df_data = {
            "Metric": metrics,
            "Black-Scholes": [bs_put, bs_pd, bs_pg, bs_pt, bs_pv, bs_pr],
            f"Binomial (N={n_comp})": [bi_put, bi_pd, bi_pg, bi_pt, bi_pv, bi_pr],
            f"Monte Carlo (Sims={sims_comp})": [mc_put, mc_pd, mc_pg, mc_pt, mc_pv, mc_pr],
        }
        put_df = pd.DataFrame(put_df_data).set_index("Metric")

        st.subheader("Call Option Comparison")
        st.dataframe(call_df, use_container_width=True)

        st.subheader("Put Option Comparison")
        st.dataframe(put_df, use_container_width=True)
