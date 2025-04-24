import math

class MarketMaker:
    def __init__(self, order_book, fair_price, base_spread=2, base_order_size=5, inventory_limit=20, maker_tag="MM1"):
        self.order_book = order_book
        self.fair_price = fair_price
        self.base_spread = base_spread
        self.base_order_size = base_order_size
        self.inventory = 0
        self.inventory_limit = inventory_limit
        self.maker_tag = maker_tag

    def quote(self, order_flow_bias=0):
        self.cancel_stale_quotes()

        # Inventory mean-reversion adjustment
        inventory_bias = -(self.inventory / self.inventory_limit) * 1.0  # Scale (tuneable)

        total_bias = order_flow_bias + inventory_bias

        if self.inventory >= self.inventory_limit:
            print("⚠️ Inventory critical high! Only quoting sell side.")
            self.place_ask(total_bias)
        elif self.inventory <= -self.inventory_limit:
            print("⚠️ Inventory critical low! Only quoting buy side.")
            self.place_bid(total_bias)
        else:
            self.place_bid(total_bias)
            self.place_ask(total_bias)

    def _dynamic_spread(self):
        """Spread increases non-linearly with inventory pressure."""
        pressure = abs(self.inventory) / self.inventory_limit
        return self.base_spread + (pressure ** 2) * 5  # Quadratic spread growth

    def _dynamic_order_size(self):
        """Quote size decreases non-linearly with inventory pressure."""
        pressure = abs(self.inventory) / self.inventory_limit
        return max(1, int(self.base_order_size * math.exp(-3 * pressure)))  # Exponential decay

    def place_bid(self, order_flow_bias=0):
        if self.inventory < self.inventory_limit:
            spread = self._dynamic_spread()
            size = self._dynamic_order_size()
            bid_price = self.fair_price - spread + order_flow_bias  # Apply skew
            self.order_book.add_order("buy", bid_price, size,self.maker_tag)

    def place_ask(self, order_flow_bias=0):
        if self.inventory > -self.inventory_limit:
            spread = self._dynamic_spread()
            size = self._dynamic_order_size()
            ask_price = self.fair_price + spread + order_flow_bias  # Apply skew
            self.order_book.add_order("sell", ask_price, size)

    def cancel_stale_quotes(self):
        self.order_book.bids = [o for o in self.order_book.bids if abs(o["price"] - self.fair_price) > 10]
        self.order_book.asks = [o for o in self.order_book.asks if abs(o["price"] - self.fair_price) > 10]

    def process_trades(self, trades, side):
        if trades:
            for trade in trades:
                maker = trade.get("maker")
                if maker == "MM1":
                    self.market_maker.process_trades([trade], side)
                    self.evaluator.record_trade([trade], side)
                elif maker == "MM2":
                    self.market_maker2.process_trades([trade], side)
                    self.evaluator2.record_trade([trade], side)

