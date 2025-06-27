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

def bs_greeks(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S*sigma*np.sqrt(T))
    theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)
    vega = S * norm.pdf(d1) * np.sqrt(T)
    rho = K * T * np.exp(-r*T) * norm.cdf(d2)
    return delta, gamma, theta, vega, rho

# ------------------- Binomial Option Pricing -------------------
def binomial_option_pricing(S, K, T, r, sigma, option_type="call", N=100):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # Initialize asset prices at maturity
    prices = S * (u ** (np.arange(N, -1, -1))) * (d ** (np.arange(0, N + 1, 1)))

    # Initialize option values at maturity
    if option_type == "call":
        option_values = np.maximum(0, prices - K)
    else:
        option_values = np.maximum(0, K - prices)

    # Backward induction
    for i in range(N - 1, -1, -1):
        option_values = np.exp(-r * dt) * (p * option_values[1:] + (1 - p) * option_values[:-1])
    
    return option_values[0]

def binomial_greeks(S, K, T, r, sigma, option_type="call", N=100, delta_S=0.01, delta_sigma=0.001, delta_T=0.001, delta_r=0.0001):
    # Delta
    price_up = binomial_option_pricing(S + delta_S, K, T, r, sigma, option_type, N)
    price_down = binomial_option_pricing(S - delta_S, K, T, r, sigma, option_type, N)
    delta = (price_up - price_down) / (2 * delta_S)

    # Gamma
    price_up_up = binomial_option_pricing(S + 2*delta_S, K, T, r, sigma, option_type, N)
    price_mid = binomial_option_pricing(S, K, T, r, sigma, option_type, N)
    price_down_down = binomial_option_pricing(S - 2*delta_S, K, T, r, sigma, option_type, N)
    gamma = (price_up + price_down - 2 * price_mid) / (delta_S ** 2) # Approximation

    # Vega
    price_vol_up = binomial_option_pricing(S, K, T, r, sigma + delta_sigma, option_type, N)
    price_vol_down = binomial_option_pricing(S, K, T, r, sigma - delta_sigma, option_type, N)
    vega = (price_vol_up - price_vol_down) / (2 * delta_sigma)

    # Theta
    price_t_down = binomial_option_pricing(S, K, T - delta_T, r, sigma, option_type, N)
    theta = (price_t_down - binomial_option_pricing(S, K, T, r, sigma, option_type, N)) / delta_T # Approximation

    # Rho
    price_r_up = binomial_option_pricing(S, K, T, r + delta_r, sigma, option_type, N)
    price_r_down = binomial_option_pricing(S, K, T, r - delta_r, sigma, option_type, N)
    rho = (price_r_up - price_r_down) / (2 * delta_r)

    return delta, gamma, theta, vega, rho


# ------------------- Monte Carlo Simulation -------------------
def monte_carlo_option_pricing(S, K, T, r, sigma, option_type="call", num_simulations=10000):
    dt = T
    
    # Generate random standard normal variables
    z = np.random.standard_normal(num_simulations)
    
    # Simulate stock prices at maturity
    ST = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    
    # Calculate option payoffs
    if option_type == "call":
        payoffs = np.maximum(0, ST - K)
    else:
        payoffs = np.maximum(0, K - ST)
        
    # Discount back to present value
    option_price = np.exp(-r * dt) * np.mean(payoffs)
    return option_price

def mc_greeks(S, K, T, r, sigma, option_type="call", num_simulations=10000, delta=0.01):
    # Delta (finite difference)
    price_up = monte_carlo_option_pricing(S + delta, K, T, r, sigma, option_type, num_simulations)
    price_down = monte_carlo_option_pricing(S - delta, K, T, r, sigma, option_type, num_simulations)
    delta_val = (price_up - price_down) / (2 * delta)

    # Gamma (finite difference)
    price_up_up = monte_carlo_option_pricing(S + 2*delta, K, T, r, sigma, option_type, num_simulations)
    price_mid = monte_carlo_option_pricing(S, K, T, r, sigma, option_type, num_simulations)
    price_down_down = monte_carlo_option_pricing(S - 2*delta, K, T, r, sigma, option_type, num_simulations)
    gamma_val = (price_up + price_down - 2 * price_mid) / (delta ** 2)

    # Vega (finite difference)
    price_sigma_up = monte_carlo_option_pricing(S, K, T, r, sigma + delta, option_type, num_simulations)
    price_sigma_down = monte_carlo_option_pricing(S, K, T, r, sigma - delta, option_type, num_simulations)
    vega_val = (price_sigma_up - price_sigma_down) / (2 * delta)

    # Theta (finite difference)
    price_T_down = monte_carlo_option_pricing(S, K, T - delta, r, sigma, option_type, num_simulations)
    theta_val = (price_T_down - monte_carlo_option_pricing(S, K, T, r, sigma, option_type, num_simulations)) / delta

    # Rho (finite difference)
    price_r_up = monte_carlo_option_pricing(S, K, T, r + delta, sigma, option_type, num_simulations)
    price_r_down = monte_carlo_option_pricing(S, K, T, r - delta, sigma, option_type, num_simulations)
    rho_val = (price_r_up - price_r_down) / (2 * delta)
    
    return delta_val, gamma_val, theta_val, vega_val, rho_val


# ------------------- Sidebar Controls -------------------
st.sidebar.markdown("## 🔧 Configure Parameters")

selected_model = st.sidebar.selectbox("Select Pricing Model", ["Black-Scholes", "Binomial Option Pricing", "Monte Carlo Simulation"])

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

    if selected_model == "Binomial Option Pricing":
        N_binomial = st.slider("Number of Steps (N) for Binomial", min_value=10, max_value=500, value=100, step=10)
    elif selected_model == "Monte Carlo Simulation":
        num_simulations_mc = st.slider("Number of Simulations for Monte Carlo", min_value=1000, max_value=100000, value=10000, step=1000)


# ------------------- Heatmap Parameters -------------------
with st.sidebar.expander("🔥 Heatmap Parameters"):
    # Auto-adjust spot price range ±25%
    auto_min_spot = round(spot_price * 0.75, 2)
    auto_max_spot = round(spot_price * 1.25, 2)

    # Auto-adjust volatility range ±15% absolute (bounded to [0.01, 1.0])
    auto_min_vol = max(0.01, round(vol_est - 0.15, 2))
    auto_max_vol = min(1.0, round(vol_est + 0.15, 2))

    min_spot = st.number_input("Min Spot Price", value=auto_min_spot, key="heat_min_spot")
    max_spot = st.number_input("Max Spot Price", value=auto_max_spot, key="heat_max_spot")
    min_vol = st.number_input("Min Volatility", min_value=0.01, max_value=1.0,
                              value=auto_min_vol, step=0.01, key="heat_min_vol")
    max_vol = st.number_input("Max Volatility", min_value=0.01, max_value=1.0,
                              value=auto_max_vol, step=0.01, key="heat_max_vol")


# ------------------- Cross-Section Generator -------------------
with st.sidebar.expander("🎯 Cross-Section Generator"):
    option_type = st.selectbox("Option Type", ["Call", "Put"], key="opt_type")
    varying_param = st.selectbox("Parameter to Vary",
                                 ["Spot Price", "Strike Price", "Volatility", "Time to Maturity", "Risk-Free Rate"],
                                 key="var_param")
    y_axis_value = st.selectbox("Y-Axis Value",
                                ["Price", "Delta", "Gamma", "Theta", "Vega", "Rho"],
                                key="y_axis")


# ------------------- Function to get pricing and greeks based on selected model -------------------
def get_option_value_and_greeks(model, S, K, T, r, sigma, option_type, **kwargs):
    price = 0
    delta, gamma, theta, vega, rho = 0, 0, 0, 0, 0

    if model == "Black-Scholes":
        price = black_scholes(S, K, T, r, sigma, option_type)
        delta, gamma, theta, vega, rho = bs_greeks(S, K, T, r, sigma)
    elif model == "Binomial Option Pricing":
        N = kwargs.get('N', 100)
        price = binomial_option_pricing(S, K, T, r, sigma, option_type, N)
        delta, gamma, theta, vega, rho = binomial_greeks(S, K, T, r, sigma, option_type, N)
    elif model == "Monte Carlo Simulation":
        num_sims = kwargs.get('num_simulations', 10000)
        price = monte_carlo_option_pricing(S, K, T, r, sigma, option_type, num_sims)
        delta, gamma, theta, vega, rho = mc_greeks(S, K, T, r, sigma, option_type, num_sims)
    
    return price, delta, gamma, theta, vega, rho


# ------------------- Tabs -------------------
tab0, tab1, tab2, tab3 = st.tabs([
    "📋 Option Summary Table", 
    "📈 3D Graphs", 
    "🔥 Heatmaps", 
    "📷 Cross-Section"
])

# ------------------- Tab 0: Option Table -------------------
with tab0:
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

    model_params = {}
    if selected_model == "Binomial Option Pricing":
        model_params['N'] = N_binomial
    elif selected_model == "Monte Carlo Simulation":
        model_params['num_simulations'] = num_simulations_mc

    call_price, cd, cg, ct, cv, cr = get_option_value_and_greeks(selected_model, S, K, T, r, sigma, "call", **model_params)
    put_price, pd, pg, pt, pv, pr = get_option_value_and_greeks(selected_model, S, K, T, r, sigma, "put", **model_params)


    with call_col:
        st.markdown(f"""
        <div class='option-box'>
            <h1 style='color:#64b5f6; text-align: center; font-size: 36px;'>📘 Call Option ({selected_model})</h1>
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
            <h1 style='color:#f06292; text-align: center; font-size: 36px;'>📕 Put Option ({selected_model})</h1>
            <h3 style='font-size: 26px; margin-top: 25px; margin-bottom: 18px;'>Price: {put_price:.2f}</h3>
            <p><strong>Delta:</strong> {pd:.4f}</p>
            <p><strong>Gamma:</strong> {pg:.4f}</p>
            <p><strong>Theta:</strong> {pt:.2f}</p>
            <p><strong>Vega:</strong> {pv:.2f}</p>
            <p><strong>Rho:</strong> {pr:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        
# ------------------- Tab 1: 3D Graphs -------------------
def plot_3d(option_type, model, **kwargs):
    spot_range = np.linspace(0.5*S, 1.5*S, 50)
    vol_range = np.linspace(0.05, 0.5, 50)
    Spot, Vol = np.meshgrid(spot_range, vol_range)

    Z = np.zeros(Spot.shape)
    for i in range(Spot.shape[0]):
        for j in range(Spot.shape[1]):
            s_val = Spot[i, j]
            v_val = Vol[i, j]
            if model == "Black-Scholes":
                Z[i, j] = black_scholes(s_val, K, T, r, v_val, option_type)
            elif model == "Binomial Option Pricing":
                N = kwargs.get('N', 100)
                Z[i, j] = binomial_option_pricing(s_val, K, T, r, v_val, option_type, N)
            elif model == "Monte Carlo Simulation":
                num_sims = kwargs.get('num_simulations', 10000)
                Z[i, j] = monte_carlo_option_pricing(s_val, K, T, r, v_val, option_type, num_sims)


    fig = go.Figure(data=[go.Surface(
        x=Spot, y=Vol, z=Z, colorscale='Viridis', showscale=True)])

    fig.update_layout(
        title=f"{option_type.capitalize()} Option Price Surface ({model})",
        scene=dict(
            xaxis_title="Spot Price",
            yaxis_title="Volatility",
            zaxis_title="Option Price"
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

with tab1:
    model_params_3d = {}
    if selected_model == "Binomial Option Pricing":
        model_params_3d['N'] = N_binomial
    elif selected_model == "Monte Carlo Simulation":
        model_params_3d['num_simulations'] = num_simulations_mc

    st.plotly_chart(plot_3d("call", selected_model, **model_params_3d), use_container_width=True)
    st.plotly_chart(plot_3d("put", selected_model, **model_params_3d), use_container_width=True)

# ------------------- Tab 2: Heatmaps -------------------
def plot_heatmaps(model, **kwargs):
    spot_range = np.round(np.linspace(min_spot, max_spot, 10), 2)
    vol_range = np.round(np.linspace(min_vol, max_vol, 10), 2)

    call_prices = np.zeros((len(vol_range), len(spot_range)))
    put_prices = np.zeros((len(vol_range), len(spot_range)))

    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            if model == "Black-Scholes":
                call_prices[i, j] = black_scholes(spot, K, T, r, vol, "call")
                put_prices[i, j] = black_scholes(spot, K, T, r, vol, "put")
            elif model == "Binomial Option Pricing":
                N = kwargs.get('N', 100)
                call_prices[i, j] = binomial_option_pricing(spot, K, T, r, vol, "call", N)
                put_prices[i, j] = binomial_option_pricing(spot, K, T, r, vol, "put", N)
            elif model == "Monte Carlo Simulation":
                num_sims = kwargs.get('num_simulations', 10000)
                call_prices[i, j] = monte_carlo_option_pricing(spot, K, T, r, vol, "call", num_sims)
                put_prices[i, j] = monte_carlo_option_pricing(spot, K, T, r, vol, "put", num_sims)


    # Call Option Heatmap
    fig_call, ax_call = plt.subplots(figsize=(8, 6))
    sns.heatmap(call_prices, 
                xticklabels=[f"{s:.2f}" for s in spot_range],
                yticklabels=[f"{v:.2f}" for v in vol_range],
                annot=True, fmt=".2f", cmap="viridis", ax=ax_call)
    ax_call.set_title(f"Call Option Heatmap ({model})", fontsize=14, fontweight="bold")
    ax_call.set_xlabel("Spot Price")
    ax_call.set_ylabel("Volatility")

    # Put Option Heatmap
    fig_put, ax_put = plt.subplots(figsize=(8, 6))
    sns.heatmap(put_prices, 
                xticklabels=[f"{s:.2f}" for s in spot_range],
                yticklabels=[f"{v:.2f}" for v in vol_range],
                annot=True, fmt=".2f", cmap="viridis", ax=ax_put)
    ax_put.set_title(f"Put Option Heatmap ({model})", fontsize=14, fontweight="bold")
    ax_put.set_xlabel("Spot Price")
    ax_put.set_ylabel("Volatility")

    # Display both
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig_call)
    with col2:
        st.pyplot(fig_put)

with tab2 :
    model_params_heatmap = {}
    if selected_model == "Binomial Option Pricing":
        model_params_heatmap['N'] = N_binomial
    elif selected_model == "Monte Carlo Simulation":
        model_params_heatmap['num_simulations'] = num_simulations_mc
    plot_heatmaps(selected_model, **model_params_heatmap)

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

    model_params_cross_section = {}
    if selected_model == "Binomial Option Pricing":
        model_params_cross_section['N'] = N_binomial
    elif selected_model == "Monte Carlo Simulation":
        model_params_cross_section['num_simulations'] = num_simulations_mc

    for val in x_vals:
        temp = fixed.copy()
        temp[var_param_key] = val
        if y_axis_value == "Price":
            y, _, _, _, _, _ = get_option_value_and_greeks(selected_model, temp["S"], temp["K"], temp["T"], temp["r"], temp["sigma"], option_type, **model_params_cross_section)
        else:
            greeks_map = {"Delta": 0, "Gamma": 1, "Theta": 2, "Vega": 3, "Rho": 4}
            _, delta, gamma, theta, vega, rho = get_option_value_and_greeks(selected_model, temp["S"], temp["K"], temp["T"], temp["r"], temp["sigma"], option_type, **model_params_cross_section)
            greeks_list = [delta, gamma, theta, vega, rho]
            y = greeks_list[greeks_map[y_axis_value]]
        y_vals.append(y)

    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, label=f"{y_axis_value} vs {varying_param}")
    ax.set_xlabel(varying_param)
    ax.set_ylabel(y_axis_value)
    ax.set_title(f"{option_type.capitalize()} Option ({selected_model}): {y_axis_value} vs {varying_param}")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
