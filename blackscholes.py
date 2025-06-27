import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import yfinance as yf
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

def greeks(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S*sigma*np.sqrt(T))
    theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)
    vega = S * norm.pdf(d1) * np.sqrt(T)
    rho = K * T * np.exp(-r*T) * norm.cdf(d2)
    return delta, gamma, theta, vega, rho

# ------------------- Sidebar Inputs -------------------
st.sidebar.markdown("## 🔧 Configure Parameters")
with st.sidebar.expander("📊 Option Parameters", expanded=True):
    ticker = st.text_input("Stock Ticker (optional)", "AAPL")
    try:
        spot_price = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
    except:
        spot_price = 100.0
        st.warning("Could not fetch price. Using default.")

    S = st.number_input("Spot Price", value=spot_price, min_value=0.0)
    K = st.number_input("Strike Price", value=spot_price, min_value=0.0)
    sigma = st.number_input("Volatility (σ)", min_value=0.01, max_value=1.0, value=0.2, step=0.01)
    T = st.number_input("Time to Maturity (years)", min_value=0.01, max_value=2.0, value=0.5, step=0.01)
    r = st.number_input("Risk-Free Rate (r)", min_value=0.0, max_value=0.1, value=0.03, step=0.001)

with st.sidebar.expander("Heatmap Parameters"):
    min_spot = st.number_input("Min Spot Price", value=80.0)
    max_spot = st.number_input("Max Spot Price", value=120.0)
    min_vol = st.number_input("Min Volatility for Heatmap", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    max_vol = st.number_input("Max Volatility for Heatmap", min_value=0.01, max_value=1.0, value=0.3, step=0.01)

# ------------------- Tabs -------------------
tab0, tab1, tab2, tab3 = st.tabs([
    "📋 Option Summary Table", 
    "📈 3D Graphs", 
    "🔥 Heatmaps", 
    "📷 Cross-Section"
])

# ------------------- Tab 0: Option Table -------------------
with tab0:
    call_price = black_scholes(S, K, T, r, sigma, "call")
    put_price = black_scholes(S, K, T, r, sigma, "put")
    cd, cg, ct, cv, cr = greeks(S, K, T, r, sigma)
    pd, pg, pt, pv, pr = greeks(S, K, T, r, sigma)

    st.markdown("### Option Prices & Greeks")
    call_col, put_col = st.columns(2)

    with call_col:
        st.markdown(f"""
        <div style='background: rgba(135,206,250,0.1); padding: 20px; border-radius: 12px; box-shadow: 0 0 8px rgba(135,206,250,0.3);'>
            <h3 style='color: #64b5f6;'>📘 Call Option</h3>
            <h2 style='color: white;'>Price: {call_price:.2f}</h2>
            <p><strong>Delta:</strong> {cd:.4f}</p>
            <p><strong>Gamma:</strong> {cg:.4f}</p>
            <p><strong>Theta:</strong> {ct:.2f}</p>
            <p><strong>Vega:</strong> {cv:.2f}</p>
            <p><strong>Rho:</strong> {cr:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    with put_col:
        st.markdown(f"""
        <div style='background: rgba(255,105,180,0.1); padding: 20px; border-radius: 12px; box-shadow: 0 0 8px rgba(255,105,180,0.3);'>
            <h3 style='color: #f06292;'>📕 Put Option</h3>
            <h2 style='color: white;'>Price: {put_price:.2f}</h2>
            <p><strong>Delta:</strong> {1 - cd:.4f}</p>
            <p><strong>Gamma:</strong> {pg:.4f}</p>
            <p><strong>Theta:</strong> {pt:.2f}</p>
            <p><strong>Vega:</strong> {pv:.2f}</p>
            <p><strong>Rho:</strong> {-pr:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
        st.subheader("📕 Put Option")
        st.markdown(f"**Price:** {put_price:.2f}")
        st.markdown(f"**Delta:** {1 - cd:.4f}")
        st.markdown(f"**Gamma:** {pg:.4f}")
        st.markdown(f"**Theta:** {pt:.2f}")
        st.markdown(f"**Vega:** {pv:.2f}")
        st.markdown(f"**Rho:** {-pr:.2f}")

# ------------------- Tab 1: 3D Graphs -------------------
def plot_3d(option_type):
    spot_range = np.linspace(0.5*S, 1.5*S, 50)
    vol_range = np.linspace(0.05, 0.5, 50)
    Spot, Vol = np.meshgrid(spot_range, vol_range)
    Z = np.vectorize(black_scholes)(Spot, K, T, r, Vol, option_type)

    fig = go.Figure(data=[go.Surface(
        x=Spot, y=Vol, z=Z, colorscale='Viridis', showscale=True)])

    fig.update_layout(
        title=f"{option_type.capitalize()} Option Price Surface",
        scene=dict(
            xaxis_title="Spot Price",
            yaxis_title="Volatility",
            zaxis_title="Option Price"
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

with tab1:
    st.plotly_chart(plot_3d("call"), use_container_width=True)
    st.plotly_chart(plot_3d("put"), use_container_width=True)

# ------------------- Tab 2: Heatmaps -------------------
def plot_heatmap(option_type):
    spot_range = np.round(np.linspace(min_spot, max_spot, 10), 2)
    vol_range = np.round(np.linspace(min_vol, max_vol, 10), 2)
    Z = np.zeros((len(vol_range), len(spot_range)))

    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            Z[i][j] = black_scholes(spot, K, T, r, vol, option_type)

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = 'viridis'
    im = ax.imshow(Z, cmap=cmap, aspect='auto', origin='lower')

    ax.set_xticks(np.arange(len(spot_range)))
    ax.set_yticks(np.arange(len(vol_range)))
    ax.set_xticklabels([f"{s:.2f}" for s in spot_range], fontsize=8, rotation=45)
    ax.set_yticklabels([f"{v:.2f}" for v in vol_range], fontsize=8)

    ax.set_xlabel("Spot Price", fontsize=10)
    ax.set_ylabel("Volatility", fontsize=10)
    ax.set_title(f"{option_type.upper()}", fontsize=12, fontweight='bold')

    norm = plt.Normalize(Z.min(), Z.max())
    for i in range(len(vol_range)):
        for j in range(len(spot_range)):
            color = "black" if norm(Z[i, j]) > 0.5 else "white"
            ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center", fontsize=7, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Option Price")
    st.pyplot(fig)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        plot_heatmap("call")
    with col2:
        plot_heatmap("put")

# ------------------- Tab 3: Cross-Section -------------------
with tab3:
    slice_vol = st.number_input("Fix Volatility for Cross-Section", min_value=0.01, max_value=1.0, value=sigma, step=0.01)
    spot_range = np.linspace(0.5*S, 1.5*S, 100)
    call_prices = [black_scholes(s, K, T, r, slice_vol, "call") for s in spot_range]
    put_prices = [black_scholes(s, K, T, r, slice_vol, "put") for s in spot_range]

    fig, ax = plt.subplots()
    ax.plot(spot_range, call_prices, label="Call")
    ax.plot(spot_range, put_prices, label="Put")
    ax.set_xlabel("Spot Price")
    ax.set_ylabel("Option Price")
    ax.set_title(f"Cross-Section at Volatility = {slice_vol}")
    ax.legend()
    st.pyplot(fig)
