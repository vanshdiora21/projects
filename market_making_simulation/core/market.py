import random
import math
import time
from core.order_book import OrderBook
from core.market_maker import MarketMaker
from analysis.evaluator import Evaluator

class MarketSimulator:
    def __init__(self, mid_price=100, regime="mean-reverting"):
        self.order_book = OrderBook()
        self.mid_price = mid_price
        self.initial_mid_price = mid_price
        self.market_maker = MarketMaker(self.order_book, fair_price=mid_price)
        self.evaluator = Evaluator(fair_price=mid_price, export_filename=export_filename)
        self.regime = regime
        self.trend_bias = 0.5  # For trending regime
        self.time_step = 0     # For oscillation

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

            # 1. Update mid-price
            self.update_mid_price()
            self.market_maker.fair_price = self.mid_price  # Sync MM fair price

            # 2. Market Maker places quotes
            self.market_maker.quote()

            # 3. Liquidity shock every 5 steps
            if step % 5 == 4:
                shock_orders = self.generate_liquidity_shock()
                for order in shock_orders:
                    trades = self.order_book.match_order(order["side"], order["quantity"])
                    print(f"Shock trade: {trades}")
                    if trades:
                        self.market_maker.process_trades(trades, order["side"])
                        self.evaluator.record_trade(trades, order["side"])
            else:
                self.simulate_latency()

                order = self.generate_external_order()
                print(f"External order: {order}")

                if order["type"] == "limit":
                    self.order_book.add_order(order["side"], order["price"], order["quantity"])
                else:
                    trades = self.order_book.match_order(order["side"], order["quantity"])
                    print(f"External trades: {trades}")
                    if trades:
                        self.market_maker.process_trades(trades, order["side"])
                        self.evaluator.record_trade(trades, order["side"])

            # 4. Evaluation snapshot
            best_bid = self.order_book.get_best_bid()
            best_ask = self.order_book.get_best_ask()
            mid_price = self.mid_price
            if best_bid and best_ask:
                mid_price = (best_bid["price"] + best_ask["price"]) / 2

            self.evaluator.snapshot(mid_price, best_bid, best_ask)

            # 5. Print market state
            print(f"Mid-price: {round(self.mid_price, 2)}")
            self.order_book.print_order_book()
            print(f"Market Maker Inventory: {self.market_maker.inventory}")

            # 6. Step delay
            self.time_step += 1
            time.sleep(1)

        # Final report
        self.evaluator.report()
