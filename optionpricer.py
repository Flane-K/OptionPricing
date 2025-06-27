import streamlit as st
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from scipy.stats import norm

st.set_page_config(layout="wide", page_title="Option Pricing Visualizer")
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

with st.sidebar.expander("📈 Underlying Stock Parameters", expanded=True):
    ticker = st.text_input("Enter Stock Ticker", value="AAPL").upper()
    
    # Initialize defaults
    spot_price, vol_est, rf_fetch = 100.0, 0.20, 0.03
    currency = "$"
    
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        if not hist.empty:
            spot_price = hist["Close"].iloc[-1]
            currency = "₹" if ticker.endswith(".NS") else "$"
            st.success(f"Fetched Spot Price: {currency}{spot_price:.2f}")

            hist30 = stock.history(period="30d")["Close"]
            log_ret = np.log(hist30 / hist30.shift(1)).dropna()
            vol_est = np.std(log_ret) * np.sqrt(252)
            st.success(f"Estimated Volatility (30d): {vol_est:.2%}")
        else:
            st.warning("No price data. Using default spot price.")
    except Exception:
        st.warning("Error fetching stock data. Using defaults.")

    S = st.number_input("Spot Price", value=float(spot_price), min_value=0.01, format="%.2f")
    sigma = st.number_input("Volatility (σ)", min_value=0.01, max_value=2.0, value=round(vol_est, 2), step=0.01)

    try:
        rf_fetch = yf.Ticker("^IRX").history(period="1d")["Close"].iloc[-1] / 100
        st.success(f"Fetched Risk-Free Rate (US T-Bill): {rf_fetch:.3%}")
        if not ticker.endswith((".NS", ".BO")):
             st.info("Using US T-Bill as risk-free rate. For non-US stocks, consider entering a local rate manually.")
    except Exception:
        st.warning("Could not fetch risk-free rate — using default.")

    r = st.number_input("Risk-Free Rate (r)", min_value=0.0, max_value=0.2, value=float(rf_fetch), step=0.001, format="%.3f")

with st.sidebar.expander("⚙️ Option Parameters", expanded=True):
    K = st.number_input("Strike Price", value=float(spot_price), min_value=0.01, format="%.2f")
    T = st.number_input("Time to Maturity (yrs)", min_value=0.01, max_value=5.0, value=0.5, step=0.01)

    if selected_model == "Binomial Option Pricing":
        N_binomial = st.slider("Number of Steps (N)", min_value=10, max_value=1000, value=100, step=10)
    elif selected_model == "Monte Carlo Simulation":
        num_simulations_mc = st.slider("Number of Simulations", min_value=1000, max_value=100000, value=10000, step=1000)

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

with st.spinner(f"Calculating with {selected_model} model, please wait..."):
    call_price, cd, cg, ct, cv, cr = get_option_value_and_greeks(selected_model, S, K, T, r, sigma, "call", **model_params)
    put_price, pd, pg, pt, pv, pr = get_option_value_and_greeks(selected_model, S, K, T, r, sigma, "put", **model_params)

# ------------------- TABS -------------------
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Summary", "💸 Payoff Diagram", "📊 Model Comparison", "📈 3D Surface", "🔥 Heatmaps", "🎯 Cross-Section"
])

# ------------------- Tab 0: Option Summary -------------------
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
        gcol2.metric(label="Theta (Θ)", value=f"{pt:.4f}")
        gcol1.metric(label="Rho (Ρ)", value=f"{pr:.4f}")

# ------------------- Tab 1: Payoff Diagram -------------------
with tab1:
    st.header("Profit/Loss at Expiration")
    spot_range = np.linspace(S * 0.7, S * 1.3, 100)
    
    call_payoff = np.maximum(spot_range - K, 0) - call_price
    put_payoff = np.maximum(K - spot_range, 0) - put_price
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spot_range, y=call_payoff, mode='lines', name='Call Option P/L'))
    fig.add_trace(go.Scatter(x=spot_range, y=put_payoff, mode='lines', name='Put Option P/L'))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=K, line_dash="dash", line_color="red", name="Strike Price")

    fig.update_layout(
        title="Option Payoff Profile",
        xaxis_title="Stock Price at Expiration",
        yaxis_title="Profit / Loss per Share",
        legend_title="Option Type"
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------- Tab 2: Model Comparison -------------------
with tab2:
    st.header("Model Price Comparison")
    with st.spinner("Running all models for comparison..."):
        # Black-Scholes
        bs_call, bs_cd, bs_cg, bs_ct, bs_cv, bs_cr = get_option_value_and_greeks("Black-Scholes", S, K, T, r, sigma, "call")
        bs_put, bs_pd, bs_pg, bs_pt, bs_pv, bs_pr = get_option_value_and_greeks("Black-Scholes", S, K, T, r, sigma, "put")

        # Binomial
        n_comp = model_params.get('N', 100)
        bi_call, bi_cd, bi_cg, bi_ct, bi_cv, bi_cr = get_option_value_and_greeks("Binomial Option Pricing", S, K, T, r, sigma, "call", N=n_comp)
        bi_put, bi_pd, bi_pg, bi_pt, bi_pv, bi_pr = get_option_value_and_greeks("Binomial Option Pricing", S, K, T, r, sigma, "put", N=n_comp)

        # Monte Carlo
        sims_comp = model_params.get('num_simulations', 10000)
        mc_call, mc_cd, mc_cg, mc_ct, mc_cv, mc_cr = get_option_value_and_greeks("Monte Carlo Simulation", S, K, T, r, sigma, "call", num_simulations=sims_comp)
        mc_put, mc_pd, mc_pg, mc_pt, mc_pv, mc_pr = get_option_value_and_greeks("Monte Carlo Simulation", S, K, T, r, sigma, "put", num_simulations=sims_comp)

    st.subheader("Call Option Comparison")
    st.dataframe({
        "Metric": ["Price", "Delta", "Gamma", "Theta", "Vega", "Rho"],
        "Black-Scholes": [bs_call, bs_cd, bs_cg, bs_ct, bs_cv, bs_cr],
        "Binomial": [bi_call, bi_cd, bi_cg, bi_ct, bi_cv, bi_cr],
        "Monte Carlo": [mc_call, mc_cd, mc_cg, mc_ct, mc_cv, mc_cr],
    })
    
    st.subheader("Put Option Comparison")
    st.dataframe({
        "Metric": ["Price", "Delta", "Gamma", "Theta", "Vega", "Rho"],
        "Black-Scholes": [bs_put, bs_pd, bs_pg, bs_pt, bs_pv, bs_pr],
        "Binomial": [bi_put, bi_pd, bi_pg, bi_pt, bi_pv, bi_pr],
        "Monte Carlo": [mc_put, mc_pd, mc_pg, mc_pt, mc_pv, mc_pr],
    })

# ------------------- Tab 3: 3D Graphs -------------------
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

        fig = go.Figure(data=[go.Surface(x=Spot, y=Time, z=Z, colorscale='viridis')])
        fig.update_layout(
            title=f"{option_type.capitalize()} Option Price vs. Spot and Time",
            scene=dict(xaxis_title="Spot Price", yaxis_title="Time to Maturity", zaxis_title="Option Price"),
            margin=dict(l=0, r=0, b=0, t=40))
        return fig

    st.plotly_chart(plot_3d("call", selected_model, **model_params), use_container_width=True)
    st.plotly_chart(plot_3d("put", selected_model, **model_params), use_container_width=True)

# ------------------- Tab 4: Heatmaps -------------------
with tab4:
    st.header(f"Price Heatmaps vs. Spot & Volatility ({selected_model})")
    
    # Auto-adjust ranges
    auto_min_spot = round(S * 0.8, 2)
    auto_max_spot = round(S * 1.2, 2)
    auto_min_vol = max(0.01, round(sigma - 0.1, 2))
    auto_max_vol = min(1.0, round(sigma + 0.1, 2))
    
    with st.expander("Adjust Heatmap Parameters"):
        min_spot = st.number_input("Min Spot Price", value=auto_min_spot)
        max_spot = st.number_input("Max Spot Price", value=auto_max_spot)
        min_vol = st.number_input("Min Volatility", value=auto_min_vol, step=0.01)
        max_vol = st.number_input("Max Volatility", value=auto_max_vol, step=0.01)

    spot_range = np.linspace(min_spot, max_spot, 10)
    vol_range = np.linspace(min_vol, max_vol, 10)
    
    call_prices = np.zeros((len(vol_range), len(spot_range)))
    put_prices = np.zeros((len(vol_range), len(spot_range)))

    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            call_prices[i, j], _, _, _, _, _ = get_option_value_and_greeks(selected_model, spot, K, T, r, vol, "call", **model_params)
            put_prices[i, j], _, _, _, _, _ = get_option_value_and_greeks(selected_model, spot, K, T, r, vol, "put", **model_params)

    def plot_plotly_heatmap(prices, spot_range, vol_range, title):
        fig = go.Figure(data=go.Heatmap(
            z=prices,
            x=[f"{s:.2f}" for s in spot_range],
            y=[f"{v:.2f}" for v in vol_range],
            hoverongaps=False,
            colorscale='viridis',
            text=np.around(prices, 2),
            texttemplate="%{text}"
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Spot Price",
            yaxis_title="Volatility"
        )
        return fig

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_plotly_heatmap(call_prices, spot_range, vol_range, "Call Option Prices"), use_container_width=True)
    with col2:
        st.plotly_chart(plot_plotly_heatmap(put_prices, spot_range, vol_range, "Put Option Prices"), use_container_width=True)

# ------------------- Tab 5: Cross-Section -------------------
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
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines'))
    fig.update_layout(
        title=f"{option_type_cs} {y_axis_value} vs. {varying_param} ({selected_model})",
        xaxis_title=varying_param,
        yaxis_title=y_axis_value
    )
    st.plotly_chart(fig, use_container_width=True)
