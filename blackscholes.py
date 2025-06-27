import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import yfinance as yf
from scipy.stats import norm

st.set_page_config(layout="wide", page_title="Vibe Coding – Option Pricing Visualizer")

def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S * norm.cdf(-d1)

st.sidebar.title("📊 Option Parameters")
ticker = st.sidebar.text_input("Stock Ticker (optional)", "AAPL")
try:
    spot_price = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
except:
    spot_price = 100.0
    st.sidebar.warning("Could not fetch price. Using default.")

S = st.sidebar.slider("Spot Price", 0.5*spot_price, 1.5*spot_price, spot_price)
K = st.sidebar.slider("Strike Price", 0.5*spot_price, 1.5*spot_price, spot_price)
sigma = st.sidebar.slider("Volatility (σ)", 0.05, 1.0, 0.2)
T = st.sidebar.slider("Time to Maturity (years)", 0.01, 2.0, 0.5)
r = st.sidebar.slider("Risk-Free Rate (r)", 0.0, 0.1, 0.03)

tab1, tab2, tab3 = st.tabs(["📈 3D Graphs", "🔥 Heatmaps", "📷 Cross-Section"])

def plot_3d(option_type):
    spot_range = np.linspace(0.5*S, 1.5*S, 50)
    vol_range = np.linspace(0.05, 0.5, 50)
    Spot, Vol = np.meshgrid(spot_range, vol_range)
    Z = np.vectorize(black_scholes)(Spot, K, T, r, Vol, option_type)

    fig = go.Figure(data=[go.Surface(
        x=Spot, y=Vol, z=Z, colorscale='Cividis', showscale=True)])

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

def plot_heatmap(option_type):
    spot_range = np.linspace(0.5*S, 1.5*S, 10)
    vol_range = np.linspace(0.1, 0.3, 10)
    Z = np.zeros((len(vol_range), len(spot_range)))

    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            Z[i][j] = black_scholes(spot, K, T, r, vol, option_type)

    fig, ax = plt.subplots()
    cmap = "plasma"  # You can try: 'cividis', 'inferno', 'viridis'
    c = ax.imshow(Z, aspect='auto', cmap=cmap,
                  extent=[spot_range[0], spot_range[-1], vol_range[0], vol_range[-1]],
                  origin='lower', interpolation='nearest')
    
    ax.set_xticks(np.round(spot_range, 2))
    ax.set_yticks(np.round(vol_range, 2))
    ax.set_xlabel("Spot Price")
    ax.set_ylabel("Volatility")
    ax.set_title(f"{option_type.upper()}")

    # Annotate values
    for i in range(len(vol_range)):
        for j in range(len(spot_range)):
            ax.text(spot_range[j], vol_range[i], f"{Z[i, j]:.2f}", color="white",
                    ha='center', va='center', fontsize=8, fontweight='bold')

    fig.colorbar(c, ax=ax)
    st.pyplot(fig)


with tab2:
    col1, col2 = st.columns(2)
    with col1:
        plot_heatmap("call")
    with col2:
        plot_heatmap("put")

with tab3:
    slice_vol = st.slider("Fix Volatility for Cross-Section", 0.05, 1.0, sigma)
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

st.sidebar.subheader("🧠 Option Greeks (Current Input)")

def greeks(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S*sigma*np.sqrt(T))
    theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)
    vega = S * norm.pdf(d1) * np.sqrt(T)
    rho = K * T * np.exp(-r*T) * norm.cdf(d2)
    return delta, gamma, theta, vega, rho

d, g, t, v, r_ = greeks(S, K, T, r, sigma)
st.sidebar.markdown(f"""
- **Delta**: {d:.4f}  
- **Gamma**: {g:.4f}  
- **Theta**: {t:.2f}  
- **Vega**: {v:.2f}  
- **Rho**: {r_:.2f}
""")
