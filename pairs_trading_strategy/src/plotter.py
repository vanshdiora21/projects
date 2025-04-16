import matplotlib.pyplot as plt
import seaborn as sns

def plot_cumulative_returns(cum_returns):
    plt.figure(figsize=(12, 5))
    plt.plot(cum_returns, label='Strategy Returns')
    plt.title("Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_trade_signals(price, signals, title='Trade Signals on Price'):
    plt.figure(figsize=(14, 6))
    plt.plot(price, label='Price', alpha=0.6)
    buy_signals = signals[signals == 1].index
    sell_signals = signals[signals == -1].index
    plt.scatter(buy_signals, price.loc[buy_signals], label='Buy', marker='^', color='green')
    plt.scatter(sell_signals, price.loc[sell_signals], label='Sell', marker='v', color='red')
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)
    plt.show()
