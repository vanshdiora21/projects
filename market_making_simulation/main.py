import json
from core.market import MarketSimulator

def load_config(path="config.json"):
    """
    Loads simulation settings from the config.json file.
    """
    with open(path, "r") as file:
        return json.load(file)

def main():
    # 1. Load configuration
    config = load_config()

    # 2. Extract simulation parameters
    steps = config["simulation_steps"]

    # 3. Initialize the market simulator with config
    market = MarketSimulator(config)

    # 4. Run the simulation
    market.simulate_order_flow(steps)

if __name__ == "__main__":
    main()
