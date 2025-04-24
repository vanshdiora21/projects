import random
import math
import time
from core.order_book import OrderBook
from core.market_maker import MarketMaker
from analysis.evaluator import Evaluator
import matplotlib.pyplot as plt

class MarketSimulator:
    def __init__(self, mid_price=100, regime="mean-reverting", export_filename="simulation_metrics.csv"):
        self.order_book = OrderBook()
        self.mid_price = mid_price
        self.initial_mid_price = mid_price
        self.market_maker = MarketMaker(self.order_book, fair_price=mid_price, inventory_limit=20, maker_tag="MM1")
        self.market_maker2 = MarketMaker(self.order_book, fair_price=mid_price, inventory_limit=10, maker_tag="MM2")


        self.evaluator = Evaluator(fair_price=mid_price, export_filename=export_filename)  # Pass export_filename here
        self.evaluator2 = Evaluator(fair_price=mid_price, export_filename="MM2_" + export_filename)
        self.regime = regime
        self.trend_bias = 0.5
        self.time_step = 0
        self.order_flow_window = []
        self.order_flow_limit = 5

    def simulate_latency(self):
        """
        Simulate random latency between external orders (100ms - 500ms).
        """
        latency = random.uniform(0.1, 0.5)
        time.sleep(latency)

    def update_mid_price(self):
        """
        Update mid-price based on the selected market regime.
        """
        if self.regime == "mean-reverting":
            # Sine wave oscillation + small random noise
            oscillation = 5 * math.sin(0.2 * self.time_step)
            noise = random.uniform(-1, 1)
            self.mid_price = self.initial_mid_price + oscillation + noise

        elif self.regime == "trending":
            # Random walk with upward or downward bias
            direction = 1 if random.random() < self.trend_bias else -1
            drift = direction * random.uniform(0, 1)
            self.mid_price += drift
    def update_order_flow(self, side):
        self.order_flow_window.append(side)
        if len(self.order_flow_window) > self.order_flow_limit:
            self.order_flow_window.pop(0)

    def generate_external_order(self):
        """
        Simulate external participants submitting orders.
        """
        order_type = random.choice(["limit", "market"])
        side = random.choice(["buy", "sell"])
        quantity = random.randint(1, 10)

        if order_type == "limit":
            price_offset = random.randint(-5, 5)
            price = self.mid_price + price_offset
            return {"type": "limit", "side": side, "price": price, "quantity": quantity}
        else:
            return {"type": "market", "side": side, "quantity": quantity}

    def generate_liquidity_shock(self):
        """
        Adaptive shock based on MM inventory.
        """
        mm_inventory = self.market_maker.inventory

        # Adaptive shock direction
        if mm_inventory > 10:
            shock_side = "sell"
        elif mm_inventory < -10:
            shock_side = "buy"
        else:
            shock_side = random.choice(["buy", "sell"])

        num_orders = random.randint(3, 7)
        shock_orders = []

        for _ in range(num_orders):
            quantity = random.randint(10, 20)
            shock_orders.append({"type": "market", "side": shock_side, "quantity": quantity})

        print(f"\n🚨 Adaptive Liquidity Shock! {num_orders} large {shock_side.upper()} market orders incoming.")
        return shock_orders

    def simulate_order_flow(self, num_steps=15):
        for step in range(num_steps):
            print(f"\n--- Step {step+1} ---")

            # 1. Update mid-price (market regime)
            self.update_mid_price()
            self.market_maker.fair_price = self.mid_price
            self.market_maker2.fair_price = self.mid_price  # Sync MM2

            # 2. Calculate order flow bias for MM1
            buy_count = self.order_flow_window.count("buy")
            sell_count = self.order_flow_window.count("sell")
            flow_bias = buy_count - sell_count
            order_flow_adjustment = flow_bias * 0.5  # Skew factor

            # 3. MM1 and MM2 quote independently
            self.market_maker.quote(order_flow_adjustment)  # MM1 with skew
            self.market_maker2.quote()                      # MM2 with no skew

            # 4. Handle liquidity shock every 5 steps
            if step % 5 == 4:
                shock_orders = self.generate_liquidity_shock()
                for order in shock_orders:
                    trades = self.order_book.match_order(order["side"], order["quantity"])
                    print(f"Shock trade: {trades}")
                    self.process_trades(trades, order["side"])
                    self.update_order_flow(order["side"])
            else:
                # 5. Simulate latency
                self.simulate_latency()

                # 6. External order arrives
                order = self.generate_external_order()
                print(f"External order: {order}")

                if order["type"] == "limit":
                    self.order_book.add_order(order["side"], order["price"], order["quantity"])
                else:  # Market order
                    trades = self.order_book.match_order(order["side"], order["quantity"])
                    print(f"External trades: {trades}")
                    self.process_trades(trades, order["side"])
                    self.update_order_flow(order["side"])

            # 7. Noise trader acts every 3 steps
            if step % 3 == 0:
                self.noise_trader_action()

            # 8. Snapshot for evaluation
            best_bid = self.order_book.get_best_bid()
            best_ask = self.order_book.get_best_ask()
            mid_price = self.mid_price
            if best_bid and best_ask:
                mid_price = (best_bid["price"] + best_ask["price"]) / 2

            self.evaluator.snapshot(mid_price, best_bid, best_ask)
            self.evaluator2.snapshot(mid_price, best_bid, best_ask)

            # 9. Optional: Plot order book depth every 5 steps
            if step % 5 == 0:
                self.plot_order_book_depth()

            # 10. Print state
            print(f"Mid-price: {round(self.mid_price, 2)}")
            self.order_book.print_order_book()
            print(f"MM1 Inventory: {self.market_maker.inventory} | MM2 Inventory: {self.market_maker2.inventory}")
            print(f"Order Flow Window: {self.order_flow_window}")

            # 11. Advance time
            self.time_step += 1
            time.sleep(1)

        # 12. Final reports
        self.evaluator.report()
        self.evaluator2.report()

    def plot_order_book_depth(self):
        

        depth = self.order_book.get_depth(bin_size=1)
        prices = sorted(depth.keys())

        buy_sizes = [depth[p]["buy"] for p in prices]
        sell_sizes = [depth[p]["sell"] for p in prices]

        plt.figure(figsize=(8, 4))
        plt.bar(prices, buy_sizes, width=0.8, label="Buy Depth", color="blue", alpha=0.6)
        plt.bar(prices, sell_sizes, width=0.8, label="Sell Depth", color="red", alpha=0.6, bottom=buy_sizes)
        plt.axvline(self.mid_price, linestyle='--', color='black', label="Mid-price")
        plt.xlabel("Price")
        plt.ylabel("Volume")
        plt.title("Order Book Depth")
        plt.legend()
        plt.show()

    def noise_trader_action(self):
        """
        Noise trader submits random market orders independent of external flow.
        """
        side = random.choice(["buy", "sell"])
        quantity = random.randint(5, 15)  # Larger orders
        print(f"🟡 Noise trader submits {side.upper()} {quantity}")

        trades = self.order_book.match_order(side, quantity)
        if trades:
            self.market_maker.process_trades(trades, side)
            self.evaluator.record_trade(trades, side)
        # Update order flow window
        self.order_flow_window.append(side)
        if len(self.order_flow_window) > self.order_flow_limit:
            self.order_flow_window.pop(0)
