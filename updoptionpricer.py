import streamlit as st
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from scipy.stats import norm
import pandas as pd

st.set_page_config(layout="wide", page_title="Option Pricing Visualizer")
st.title("📈 Option Pricing Visualizer")


# ------------------- Data for Stock Markets -------------------
# A pre-defined dictionary for markets, stocks, currency, and risk-free rates.
MARKETS = {
    "🇺🇸 US Exchanges": {
        "currency": "$",
        "rf_ticker": "^IRX",  # 13 Week Treasury Bill
        "rf_name": "US 13W T-Bill",
        "stocks": {
            "AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "GOOGL": "Alphabet Inc.",
            "AMZN": "Amazon.com, Inc.", "NVDA": "NVIDIA Corporation", "TSLA": "Tesla, Inc.",
            "JPM": "JPMorgan Chase & Co.", "V": "Visa Inc.", "JNJ": "Johnson & Johnson",
            "WMT": "Walmart Inc.", "PG": "Procter & Gamble Co.", "META": "Meta Platforms, Inc."
        }
    },
    "🇮🇳 Indian Exchanges": {
        "currency": "₹",
        "rf_ticker": "^NSITEN", # India 10Y Bond Yield
        "rf_name": "India 10Y Bond",
        "stocks": {
            "RELIANCE.NS": "Reliance Industries Limited", "TCS.NS": "Tata Consultancy Services",
            "HDFCBANK.NS": "HDFC Bank Limited", "INFY.NS": "Infosys Limited",
            "ICICIBANK.NS": "ICICI Bank Limited", "HINDUNILVR.NS": "Hindustan Unilever Limited",
            "SBIN.NS": "State Bank of India", "BHARTIARTL.NS": "Bharti Airtel Limited",
            "ITC.NS": "ITC Limited"
        }
    },
    "🇪🇺 European Exchanges": {
        "currency": "€",
        "rf_ticker": "DE10Y.SG", # German 10-Year Bond as a proxy for Eurozone rate
        "rf_name": "German 10Y Bund",
        "stocks": {
            "ASML.AS": "ASML Holding N.V.", "MC.PA": "LVMH Moët Hennessy Louis Vuitton SE",
            "SAP.DE": "SAP SE", "SIE.DE": "Siemens AG", "VOW3.DE": "Volkswagen AG",
            "TTE.PA": "TotalEnergies SE", "UNA.AS": "Unilever PLC", "AIR.PA": "Airbus SE"
        }
    }
}


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
    vega = (price_vol_up - price_mid) / d_sigma
    price_t_down = monte_carlo_option_pricing(S, K, T - dT, r, sigma, option_type, num_simulations)
    theta = (price_t_down - price_mid) / dT
    price_r_up = monte_carlo_option_pricing(S, K, T, r + d_r, sigma, option_type, num_simulations)
    rho = (price_r_up - price_mid) / d_r

    return delta, gamma, theta, vega, rho


# ------------------- Sidebar Controls -------------------
st.sidebar.markdown("## 🔧 Configure Parameters")
selected_model = st.sidebar.selectbox("Select Pricing Model", ["Black-Scholes", "Binomial Option Pricing", "Monte Carlo Simulation"])

if selected_model == "Binomial Option Pricing":
    N_binomial = st.sidebar.slider("Number of Steps (N)", min_value=10, max_value=1000, value=100, step=10)
elif selected_model == "Monte Carlo Simulation":
    num_simulations_mc = st.sidebar.slider("Number of Simulations", min_value=1000, max_value=100000, value=10000, step=1000)

with st.sidebar.expander("📈 Underlying Stock Parameters", expanded=True):
    selected_market_name = st.selectbox("Select Market/Stock Exchange", options=list(MARKETS.keys()))
    market_data = MARKETS[selected_market_name]
    market_stocks = market_data["stocks"]
    currency = market_data["currency"]
    
    stock_options = ["Select a stock..."] + [f"{ticker} - {name}" for ticker, name in market_stocks.items()]
    selected_stock_formatted = st.selectbox("Search for Stock (Ticker or Name)", options=stock_options, index=0)

    ticker, company_name = None, None
    spot_price, vol_est, rf_fetch = 100.0, 0.20, 0.03
    spot_help_text = "Select a stock to fetch live data."
    vol_help_text = "Volatility is estimated from the last 30 days of historical data."
    rf_help_text = "Risk-free rate is fetched based on the stock's market."
    
    if selected_stock_formatted != "Select a stock...":
        ticker = selected_stock_formatted.split(" - ")[0]
        company_name = selected_stock_formatted.split(" - ")[1]

        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1mo")
            if not hist.empty:
                spot_price = hist["Close"].iloc[-1]
                spot_help_text = f"{company_name} ({ticker}) | Last Close: {currency}{spot_price:.2f}"
                log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
                vol_est = np.std(log_ret) * np.sqrt(252)
                vol_help_text = f"Estimated Volatility (30d Ann.): {vol_est:.2%}"
            else:
                spot_help_text = f"Could not find data for '{ticker}'. Using default values."
                vol_help_text = "Could not estimate volatility. Using default value."

        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")
            spot_help_text = f"Error fetching stock data. Using defaults."
            vol_help_text = "Error fetching volatility. Using default."

        rf_ticker, rf_name = market_data["rf_ticker"], market_data["rf_name"]
        try:
            rf_data = yf.Ticker(rf_ticker).history(period="5d")
            if not rf_data.empty:
                rf_fetch = rf_data["Close"].iloc[-1] / 100
                rf_help_text = f"Fetched {rf_name} rate: {rf_fetch:.3%}"
            else:
                 rf_help_text = f"Could not fetch {rf_name} rate. Using default."
        except Exception:
            rf_help_text = f"Error fetching {rf_name} rate. Using default."

    S = st.number_input(f"Spot Price ({currency})", value=float(spot_price), min_value=0.01, format="%.2f", help=spot_help_text)
    sigma = st.number_input("Volatility (σ)", min_value=0.0, max_value=2.0, value=round(vol_est, 2), step=0.01, help=vol_help_text)
    r = st.number_input("Risk-Free Rate (r)", min_value=0.0, max_value=0.2, value=float(rf_fetch), step=0.001, format="%.3f", help=rf_help_text)

with st.sidebar.expander("⚙️ Option Parameters", expanded=True):
    K = st.number_input(f"Strike Price ({currency})", value=float(spot_price), min_value=0.01, format="%.2f")
    T = st.number_input("Time to Maturity (yrs)", min_value=0.0, max_value=5.0, value=0.5, step=0.01)

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

if not ticker:
    st.warning("Please select a stock from the sidebar to begin calculations.")
else:
    model_params = {}
    if selected_model == "Binomial Option Pricing":
        model_params['N'] = N_binomial
    elif selected_model == "Monte Carlo Simulation":
        model_params['num_simulations'] = num_simulations_mc

    with st.spinner(f"Calculating with {selected_model} model, please wait..."):
        call_price, cd, cg, ct, cv, cr = get_option_value_and_greeks(selected_model, S, K, T, r, sigma, "call", **model_params)
        put_price, pd, pg, pt, pv, pr = get_option_value_and_greeks(selected_model, S, K, T, r, sigma, "put", **model_params)

    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Summary", "💸 Payoff Diagram", "📊 Model Comparison", "📈 3D Surface", "🔥 Heatmaps", "🎯 Cross-Section"
    ])

    with tab0:
        st.header(f"Option Valuation for {company_name} ({selected_model})")
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
        fig.update_layout(title="Option Payoff Profile", xaxis_title=f"Stock Price at Expiration ({currency})", yaxis_title=f"Profit / Loss per Share ({currency})", legend_title="Option Type")
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
        
        metrics = ["Price", "Delta", "Gamma", "Theta", "Vega", "Rho"]
        
        # Call data
        call_df_data = {
            "Metric": metrics,
            "Black-Scholes": [bs_call, bs_cd, bs_cg, bs_ct, bs_cv, bs_cr],
            f"Binomial (N={n_comp})": [bi_call, bi_cd, bi_cg, bi_ct, bi_cv, bi_cr],
            f"Monte Carlo (Sims={sims_comp})": [mc_call, mc_cd, mc_cg, mc_ct, mc_cv, mc_cr]
        }
        call_df = pd.DataFrame(call_df_data).set_index("Metric")

        # Put data
        put_df_data = {
            "Metric": metrics,
            "Black-Scholes": [bs_put, bs_pd, bs_pg, bs_pt, bs_pv, bs_pr],
            f"Binomial (N={n_comp})": [bi_put, bi_pd, bi_pg, bi_pt, bi_pv, bi_pr],
            f"Monte Carlo (Sims={sims_comp})": [mc_put, mc_pd, mc_pg, mc_pt, mc_pv, mc_pr]
        }
        put_df = pd.DataFrame(put_df_data).set_index("Metric")

        st.subheader("Call Option Comparison")
        st.dataframe(call_df, use_container_width=True)
        st.subheader("Put Option Comparison")
        st.dataframe(put_df, use_container_width=True)

    with tab3:
        st.header(f"3D Price Surface ({selected_model})")
        def plot_3d(option_type, model, **kwargs):
            spot_range = np.linspace(0.5*S, 1.5*S, 30)
            time_range = np.linspace(max(0.01, T), 0.01, 30)
            Spot, Time = np.meshgrid(spot_range, time_range)
            Z = np.zeros_like(Spot)
            for i in range(Spot.shape[0]):
                for j in range(Spot.shape[1]):
                    Z[i, j], _, _, _, _, _ = get_option_value_and_greeks(model, Spot[i, j], K, Time[i, j], r, sigma, option_type.lower(), **kwargs)
            fig = go.Figure(data=[go.Surface(x=Spot, y=Time, z=Z, colorscale='viridis')])
            fig.update_layout(title=f"{option_type.capitalize()} Option Price vs. Spot and Time", scene=dict(xaxis_title="Spot Price", yaxis_title="Time to Maturity", zaxis_title="Option Price"), margin=dict(l=0, r=0, b=0, t=40))
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
        spot_range = np.linspace(min_spot, max_spot, 10)
        vol_range = np.linspace(min_vol, max_vol, 10)
        call_prices = np.zeros((len(vol_range), len(spot_range)))
        put_prices = np.zeros((len(vol_range), len(spot_range)))
        for i, vol in enumerate(vol_range):
            for j, spot in enumerate(spot_range):
                call_prices[i, j], _, _, _, _, _ = get_option_value_and_greeks(selected_model, spot, K, T, r, vol, "call", **model_params)
                put_prices[i, j], _, _, _, _, _ = get_option_value_and_greeks(selected_model, spot, K, T, r, vol, "put", **model_params)
        def plot_plotly_heatmap(prices, spot_range, vol_range, title):
            fig = go.Figure(data=go.Heatmap(z=prices, x=[f"{s:.2f}" for s in spot_range], y=[f"{v:.2f}" for v in vol_range], hoverongaps=False, colorscale='viridis', text=np.around(prices, 2), texttemplate="%{text}"))
            fig.update_layout(title=title, xaxis_title="Spot Price", yaxis_title="Volatility")
            return fig
        col1, col2 = st.columns(2)
        with col1: st.plotly_chart(plot_plotly_heatmap(call_prices, spot_range, vol_range, "Call Option Prices"), use_container_width=True)
        with col2: st.plotly_chart(plot_plotly_heatmap(put_prices, spot_range, vol_range, "Put Option Prices"), use_container_width=True)
    
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
                if var_param_key == 'T' and val <= 0: continue # Avoid error on x-axis
                if var_param_key == 'sigma' and val <= 0: continue
                price, delta, gamma, theta, vega, rho = get_option_value_and_greeks(selected_model, temp["S"], temp["K"], temp["T"], temp["r"], temp["sigma"], option_type_cs.lower(), **model_params)
                greeks_map = {"Price": price, "Delta": delta, "Gamma": gamma, "Theta": theta, "Vega": vega, "Rho": rho}
                y_vals.append(greeks_map[y_axis_value])
        fig = go.Figure()
        valid_x_vals = [x for x in x_vals if x > 0] if var_param_key in ['T', 'sigma'] else x_vals
        fig.add_trace(go.Scatter(x=valid_x_vals, y=y_vals, mode='lines'))
        fig.update_layout(title=f"{option_type_cs} {y_axis_value} vs. {varying_param} ({selected_model})", xaxis_title=varying_param, yaxis_title=y_axis_value)
        st.plotly_chart(fig, use_container_width=True)
