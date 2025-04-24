class MarketMaker:
    def __init__(self, order_book, fair_price, base_spread=2, base_order_size=5, inventory_limit=20, maker_tag="MM1"):
        self.order_book = order_book
        self.fair_price = fair_price
        self.base_spread = base_spread
        self.base_order_size = base_order_size
        self.inventory_limit = inventory_limit
        self.inventory = 0
        self.maker_tag = maker_tag

    def _dynamic_spread(self):
        return self.base_spread

    def _dynamic_order_size(self):
        return self.base_order_size

    def quote(self, order_flow_bias=0):
        self.cancel_stale_quotes()
        inventory_bias = -(self.inventory / self.inventory_limit) * 1.0
        total_bias = order_flow_bias + inventory_bias
        self.place_bid(total_bias)
        self.place_ask(total_bias)

    def place_bid(self, total_bias=0):
        if self.inventory < self.inventory_limit:
            bid_price = self.fair_price - self._dynamic_spread() + total_bias
            size = self._dynamic_order_size()
            self.order_book.add_order("buy", bid_price, size, maker=self.maker_tag)

    def place_ask(self, total_bias=0):
        if self.inventory > -self.inventory_limit:
            ask_price = self.fair_price + self._dynamic_spread() + total_bias
            size = self._dynamic_order_size()
            self.order_book.add_order("sell", ask_price, size, maker=self.maker_tag)

    def cancel_stale_quotes(self):
        self.order_book.bids = [order for order in self.order_book.bids if order.get("maker") != self.maker_tag]
        self.order_book.asks = [order for order in self.order_book.asks if order.get("maker") != self.maker_tag]

    def process_trades(self, trades, side):
        for trade in trades:
            if side == "buy":
                self.inventory -= trade["quantity"]
            elif side == "sell":
                self.inventory += trade["quantity"]