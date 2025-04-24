# main.py

from core.market import MarketSimulator

def run_simulation(regime: str, filename: str):
    print(f"\n🔁 Running simulation for regime: {regime}")
    market = MarketSimulator(mid_price=100, regime=regime)
    market.evaluator.export_filename = filename
    market.simulate_order_flow(num_steps=30)

def main():
    run_simulation(regime="mean-reverting", filename="mean_reverting_metrics.csv")
    run_simulation(regime="trending", filename="trending_metrics.csv")

if __name__ == "__main__":
    main()
