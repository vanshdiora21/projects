import json
import argparse
from core.market import MarketSimulator

def load_config(path="config.json"):
    with open(path, "r") as file:
        return json.load(file)

def override_config(config, args):
    if args.regime:
        config["regime"] = args.regime
    if args.steps:
        config["simulation_steps"] = args.steps
    if args.mm1_limit:
        config["mm1"]["inventory_limit"] = args.mm1_limit
    if args.mm2_limit:
        config["mm2"]["inventory_limit"] = args.mm2_limit
    return config

def main():
    # CLI parser
    parser = argparse.ArgumentParser(description="Market-Making Simulation CLI")
    parser.add_argument("--regime", choices=["mean-reverting", "trending"], help="Market regime type")
    parser.add_argument("--steps", type=int, help="Number of simulation steps")
    parser.add_argument("--mm1_limit", type=int, help="Inventory limit for MM1")
    parser.add_argument("--mm2_limit", type=int, help="Inventory limit for MM2")
    args = parser.parse_args()

    # Load and override config
    config = load_config()
    config = override_config(config, args)

    # Initialize and run simulation
    market = MarketSimulator(config)
    market.simulate_order_flow(config["simulation_steps"])

if __name__ == "__main__":
    main()
