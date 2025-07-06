# 📈 Option Pricing Visualizer

This Streamlit web app provides an interactive visualization of Call and Put option pricing using the Black-Scholes model. It includes 3D surface plots, heatmaps, and customizable cross-sectional charts to explore how option prices and Greeks respond to market parameters.

## 🚀 Features

- **Option Summary Table**: View Call and Put prices along with Greeks (Delta, Gamma, Theta, Vega, Rho).
- **3D Graphs**: Interactive 3D surfaces for Call and Put options vs. Spot Price and Volatility.
- **Heatmaps**: Intuitive heatmaps showing option pricing across Spot Price and Volatility.
- **Cross-Section Generator**: Generate 2D plots by varying one input parameter and tracking impact on price or Greeks.
- **Dynamic Inputs**: All calculations update live based on sidebar inputs.
- **📈 Live Stock Data**: Automatically fetches the latest stock price and historical volatility using `yfinance`.

## 📦 Dependencies

Make sure to install the required Python libraries:

```bash
pip install -r requirements.txt
```

## 🔧 How to Run

```bash
streamlit run vibe_coding.py
```


## 🌐 Live Demo

If hosted, access here: [https://optionpricing-flane.streamlit.app](https://optionpricing-flane.streamlit.app)


Built using Streamlit, NumPy, Plotly, and Matplotlib.
