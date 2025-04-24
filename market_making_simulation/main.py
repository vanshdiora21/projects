
from core.market import MarketSimulator

def main():
    # Choose between "mean-reverting" or "trending"
    market = MarketSimulator(mid_price=100, regime="mean-reverting")
    market.simulate_order_flow(num_steps=15)

if __name__ == "__main__":
    main()
