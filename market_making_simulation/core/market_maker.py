import math

class MarketMaker:
    def __init__(self, order_book, fair_price, base_spread=2, base_order_size=5, inventory_limit=20):
        self.order_book = order_book
        self.fair_price = fair_price
        self.base_spread = base_spread
        self.base_order_size = base_order_size
        self.inventory = 0
        self.inventory_limit = inventory_limit

    def quote(self):
        self.cancel_stale_quotes()

        if self.inventory >= self.inventory_limit:
            print("⚠️ Inventory critical high! Only quoting sell side.")
            self.place_ask()
        elif self.inventory <= -self.inventory_limit:
            print("⚠️ Inventory critical low! Only quoting buy side.")
            self.place_bid()
        else:
            self.place_bid()
            self.place_ask()

    def _dynamic_spread(self):
        """Spread increases non-linearly with inventory pressure."""
        pressure = abs(self.inventory) / self.inventory_limit
        return self.base_spread + (pressure ** 2) * 5  # Quadratic spread growth

    def _dynamic_order_size(self):
        """Quote size decreases non-linearly with inventory pressure."""
        pressure = abs(self.inventory) / self.inventory_limit
        return max(1, int(self.base_order_size * math.exp(-3 * pressure)))  # Exponential decay

    def place_bid(self):
        if self.inventory < self.inventory_limit:
            spread = self._dynamic_spread()
            size = self._dynamic_order_size()
            bid_price = self.fair_price - spread
            self.order_book.add_order("buy", bid_price, size)

    def place_ask(self):
        if self.inventory > -self.inventory_limit:
            spread = self._dynamic_spread()
            size = self._dynamic_order_size()
            ask_price = self.fair_price + spread
            self.order_book.add_order("sell", ask_price, size)

    def cancel_stale_quotes(self):
        self.order_book.bids = [o for o in self.order_book.bids if abs(o["price"] - self.fair_price) > 10]
        self.order_book.asks = [o for o in self.order_book.asks if abs(o["price"] - self.fair_price) > 10]

    def process_trades(self, trades, side):
        for trade in trades:
            if side == "buy":
                self.inventory += trade["quantity"]
            elif side == "sell":
                self.inventory -= trade["quantity"]
