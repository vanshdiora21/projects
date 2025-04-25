import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load simulation results
def load_metrics(file):
    df = pd.read_csv(file)
    return df

# Plot PnL Attribution
def plot_pnl(df, title):
    plt.figure(figsize=(10, 5))
    plt.plot(df['Step'], df['Realized_PnL'], label='Realized PnL', color='green')
    plt.plot(df['Step'], df['Unrealized_PnL'], label='Unrealized PnL', color='orange')
    plt.plot(df['Step'], df['Total_PnL'], label='Total PnL', color='blue')
    plt.xlabel('Simulation Step')
    plt.ylabel('PnL')
    plt.title(title)
    plt.legend()
    st.pyplot(plt)

# Plot Inventory Levels
def plot_inventory(df, title):
    plt.figure(figsize=(10, 3))
    plt.bar(df['Step'], df['Inventory'], color='gray', alpha=0.5)
    plt.xlabel('Simulation Step')
    plt.ylabel('Inventory')
    plt.title(title)
    st.pyplot(plt)

# Plot Market Impact
def plot_market_impact(file, title):
    impact_df = pd.read_csv(file)
    plt.figure(figsize=(8, 5))
    plt.scatter(impact_df['Order_Size'], impact_df['Price_Impact'], alpha=0.5)
    plt.xlabel('Order Size')
    plt.ylabel('Price Impact')
    plt.title(title)
    st.pyplot(plt)

# Streamlit layout
st.title("Market Making Simulation Dashboard")

# Sidebar configuration
st.sidebar.header("Simulation Settings")
mm_choice = st.sidebar.selectbox("Select Market Maker", ["MM1", "MM2"])

# File paths
df_file = f"exports/{mm_choice}_metrics.csv"
impact_file = f"exports/{mm_choice}_market_impact.csv"

# Load data
df = load_metrics(df_file)

# Display
st.header(f"PnL Attribution for {mm_choice}")
plot_pnl(df, f"PnL Attribution - {mm_choice}")

st.header(f"Inventory Levels for {mm_choice}")
plot_inventory(df, f"Inventory Levels - {mm_choice}")

st.header(f"Market Impact for {mm_choice}")
plot_market_impact(impact_file, f"Market Impact - {mm_choice}")
