import random
import math
import time
from core.order_book import OrderBook
from core.market_maker import MarketMaker
from analysis.evaluator import Evaluator
import matplotlib.pyplot as plt

class MarketSimulator:
    def __init__(self, config):
        self.config = config
        self.order_book = OrderBook()
        self.mid_price = 100
        self.initial_mid_price = 100

        # MM1 & MM2 setup
        mm1_cfg = config["mm1"]
        mm2_cfg = config["mm2"]

        self.market_maker = MarketMaker(self.order_book, fair_price=self.mid_price,
                                        inventory_limit=mm1_cfg["inventory_limit"],
                                        base_spread=mm1_cfg["base_spread"],
                                        base_order_size=mm1_cfg["base_order_size"],
                                        maker_tag="MM1")

        self.market_maker2 = MarketMaker(self.order_book, fair_price=self.mid_price,
                                         inventory_limit=mm2_cfg["inventory_limit"],
                                         base_spread=mm2_cfg["base_spread"],
                                         base_order_size=mm2_cfg["base_order_size"],
                                         maker_tag="MM2")

        self.evaluator = Evaluator(fair_price=self.mid_price, export_filename="MM1_metrics.csv")
        self.evaluator2 = Evaluator(fair_price=self.mid_price, export_filename="MM2_metrics.csv")

        self.regime = config["regime"]
        self.trend_bias = 0.5
        self.order_flow_window = []
        self.order_flow_limit = config["order_flow"]["window_size"]
        self.noise_trader_cfg = config["noise_trader"]
        self.time_step = 0

    def simulate_latency(self):
        time.sleep(random.uniform(0.1, 0.3))

    def update_mid_price(self):
        if self.regime == "mean-reverting":
            oscillation = 5 * math.sin(0.2 * self.time_step)
            noise = random.uniform(-1, 1)
            self.mid_price = self.initial_mid_price + oscillation + noise
        elif self.regime == "trending":
            direction = 1 if random.random() < self.trend_bias else -1
            drift = direction * random.uniform(0, 1)
            self.mid_price += drift

    def update_order_flow(self, side):
        self.order_flow_window.append(side)
        if len(self.order_flow_window) > self.order_flow_limit:
            self.order_flow_window.pop(0)

    def generate_external_order(self):
        order_type = random.choice(["limit", "market"])
        side = random.choice(["buy", "sell"])
        quantity = random.randint(10, 20)  # Boosted order sizes
        if order_type == "limit":
            price_offset = random.randint(-5, 5)
            price = self.mid_price + price_offset
            return {"type": "limit", "side": side, "price": price, "quantity": quantity}
        else:
            return {"type": "market", "side": side, "quantity": quantity}

    def generate_liquidity_shock(self):
        mm_inventory = self.market_maker.inventory
        shock_side = "sell" if mm_inventory > 10 else "buy" if mm_inventory < -10 else random.choice(["buy", "sell"])
        shock_orders = [{"type": "market", "side": shock_side, "quantity": random.randint(10, 20)} for _ in range(random.randint(3, 7))]
        print(f"\n🛘 Adaptive Liquidity Shock! {len(shock_orders)} large {shock_side.upper()} market orders incoming.")
        return shock_orders

    def process_trades(self, trades, side):
        for trade in trades:
            maker = trade.get("maker")
            if maker == "MM1":
                self.market_maker.process_trades([trade], side)
                self.evaluator.record_trade([trade], side)
            elif maker == "MM2":
                self.market_maker2.process_trades([trade], side)
                self.evaluator2.record_trade([trade], side)

    def simulate_order_flow(self, num_steps=15):
        for step in range(num_steps):
            print(f"\n--- Step {step+1} ---")
            self.update_mid_price()
            self.market_maker.fair_price = self.mid_price
            self.market_maker2.fair_price = self.mid_price

            flow_bias = self.order_flow_window.count("buy") - self.order_flow_window.count("sell")
            order_flow_adjustment = flow_bias * self.config["order_flow"]["bias_scaling"]

            self.market_maker.quote(order_flow_adjustment)
            self.market_maker2.quote()

            if step % 5 == 4:
                for order in self.generate_liquidity_shock():
                    pre_mid_price = self.mid_price
                    trades = self.order_book.match_order(order["side"], order["quantity"])
                    print(f"Shock trade: {trades}")
                    self.process_trades(trades, order["side"])
                    self.update_order_flow(order["side"])
                    best_bid = self.order_book.get_best_bid()
                    best_ask = self.order_book.get_best_ask()
                    post_mid_price = (best_bid["price"] + best_ask["price"]) / 2 if best_bid and best_ask else self.mid_price
                    self.evaluator.record_market_impact(order["quantity"], pre_mid_price, post_mid_price)
                    self.evaluator2.record_market_impact(order["quantity"], pre_mid_price, post_mid_price)
            else:
                self.simulate_latency()
                order = self.generate_external_order()
                pre_mid_price = self.mid_price
                print(f"External order: {order}")
                if order["type"] == "limit":
                    self.order_book.add_order(order["side"], order["price"], order["quantity"])
                else:
                    trades = self.order_book.match_order(order["side"], order["quantity"])
                    print(f"External trades: {trades}")
                    self.process_trades(trades, order["side"])
                    self.update_order_flow(order["side"])
                    best_bid = self.order_book.get_best_bid()
                    best_ask = self.order_book.get_best_ask()
                    post_mid_price = (best_bid["price"] + best_ask["price"]) / 2 if best_bid and best_ask else self.mid_price
                    self.evaluator.record_market_impact(order["quantity"], pre_mid_price, post_mid_price)
                    self.evaluator2.record_market_impact(order["quantity"], pre_mid_price, post_mid_price)

            if self.noise_trader_cfg["enabled"] and step % self.noise_trader_cfg["frequency"] == 0:
                side = random.choice(["buy", "sell"])
                quantity = random.randint(self.noise_trader_cfg["min_qty"], self.noise_trader_cfg["max_qty"])
                pre_mid_price = self.mid_price
                print(f"🟡 Noise trader submits {side.upper()} {quantity}")
                trades = self.order_book.match_order(side, quantity)
                self.process_trades(trades, side)
                self.update_order_flow(side)
                best_bid = self.order_book.get_best_bid()
                best_ask = self.order_book.get_best_ask()
                post_mid_price = (best_bid["price"] + best_ask["price"]) / 2 if best_bid and best_ask else self.mid_price
                self.evaluator.record_market_impact(quantity, pre_mid_price, post_mid_price)
                self.evaluator2.record_market_impact(quantity, pre_mid_price, post_mid_price)

            best_bid = self.order_book.get_best_bid()
            best_ask = self.order_book.get_best_ask()
            mid_price = self.mid_price
            if best_bid and best_ask:
                mid_price = (best_bid["price"] + best_ask["price"]) / 2
            self.evaluator.snapshot(mid_price, best_bid, best_ask)
            self.evaluator2.snapshot(mid_price, best_bid, best_ask)

            if step % 5 == 0:
                self.plot_order_book_depth()

            print(f"Mid-price: {round(self.mid_price, 2)}")
            self.order_book.print_order_book()
            print(f"MM1 Inventory: {self.market_maker.inventory} | MM2 Inventory: {self.market_maker2.inventory}")
            print(f"Order Flow Window: {self.order_flow_window}")
            self.time_step += 1
            time.sleep(1)

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
        side = random.choice(["buy", "sell"])
        quantity = random.randint(self.noise_trader_cfg["min_qty"], self.noise_trader_cfg["max_qty"])
        print(f"🟡 Noise trader submits {side.upper()} {quantity}")
        trades = self.order_book.match_order(side, quantity)
        self.process_trades(trades, side)
        self.update_order_flow(side)
