import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

class Evaluator:
    def __init__(self, fair_price):
        self.fair_price = fair_price
        self.realized_pnl = 0
        self.unrealized_pnl = 0
        self.inventory = 0
        self.export_filename = export_filename

        # History
        self.realized_pnl_history = []
        self.unrealized_pnl_history = []
        self.total_pnl_history = []
        self.inventory_history = []
        self.spread_history = []
        

    def record_trade(self, trades, side):
        for trade in trades:
            price = trade["price"]
            qty = trade["quantity"]
            if side == "buy":
                self.inventory += qty
                self.realized_pnl -= price * qty
            elif side == "sell":
                self.inventory -= qty
                self.realized_pnl += price * qty

    def update_unrealized_pnl(self, current_price):
        self.unrealized_pnl = self.inventory * current_price

    def snapshot(self, current_price, best_bid, best_ask):
        self.update_unrealized_pnl(current_price)
        total_pnl = self.realized_pnl + self.unrealized_pnl

        self.realized_pnl_history.append(self.realized_pnl)
        self.unrealized_pnl_history.append(self.unrealized_pnl)
        self.total_pnl_history.append(total_pnl)
        self.inventory_history.append(self.inventory)

        if best_bid and best_ask:
            spread = best_ask["price"] - best_bid["price"]
            self.spread_history.append(spread)
        else:
            self.spread_history.append(None)

    def report(self):
        print("\n--- Final Evaluation ---")
        print(f"Realized PnL: {self.realized_pnl}")
        print(f"Unrealized PnL: {self.unrealized_pnl}")
        print(f"Total PnL: {self.realized_pnl + self.unrealized_pnl}")
        print(f"Final Inventory: {self.inventory}")

        self.plot_metrics()
        self.export_to_csv()
        self.compute_risk_metrics() 

    def plot_metrics(self):
        plt.figure(figsize=(15, 5))

        # PnL Attribution Plot
        plt.subplot(1, 3, 1)
        plt.plot(self.realized_pnl_history, label="Realized PnL")
        plt.plot(self.unrealized_pnl_history, label="Unrealized PnL")
        plt.plot(self.total_pnl_history, label="Total PnL", linestyle='--', color='black')
        plt.title("PnL Attribution Over Time")
        plt.xlabel("Step")
        plt.ylabel("PnL")
        plt.legend()

        # Inventory Plot
        plt.subplot(1, 3, 2)
        plt.plot(self.inventory_history, label="Inventory", color='orange')
        plt.title("Inventory Over Time")
        plt.xlabel("Step")
        plt.ylabel("Inventory")
        plt.legend()

        # Spread Plot
        plt.subplot(1, 3, 3)
        plt.plot(self.spread_history, label="Spread", color='green')
        plt.title("Bid-Ask Spread Over Time")
        plt.xlabel("Step")
        plt.ylabel("Spread")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def export_to_csv(self):
        data = {
            "Step": list(range(len(self.total_pnl_history))),
            "Realized_PnL": self.realized_pnl_history,
            "Unrealized_PnL": self.unrealized_pnl_history,
            "Total_PnL": self.total_pnl_history,
            "Inventory": self.inventory_history,
            "Spread": self.spread_history
        }

        df = pd.DataFrame(data)
        os.makedirs("exports", exist_ok=True)
        path = os.path.join("exports", self.export_filename)
        df.to_csv(path, index=False)
        print(f"📁 Exported simulation metrics to: {path}")


    def compute_risk_metrics(self):
        print("\n--- Risk Metrics ---")

        # --- Max Drawdown ---
        total_pnl = np.array(self.total_pnl_history)
        peak = np.maximum.accumulate(total_pnl)
        drawdown = peak - total_pnl
        max_drawdown = np.max(drawdown)
        print(f"Max Drawdown: {max_drawdown:.2f}")

        # --- Inventory Variance ---
        inventory_var = np.var(self.inventory_history)
        print(f"Inventory Variance: {inventory_var:.2f}")

        # --- Sharpe Ratio ---
        pnl_changes = np.diff(total_pnl)
        if len(pnl_changes) > 1 and np.std(pnl_changes) > 0:
            sharpe = np.mean(pnl_changes) / np.std(pnl_changes)
            print(f"Sharpe Ratio: {sharpe:.2f}")
        else:
            print("Sharpe Ratio: N/A (not enough variance)")