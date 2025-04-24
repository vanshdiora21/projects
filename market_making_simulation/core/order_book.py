class OrderBook:
    def __init__(self):
        self.bids = []  # Buy orders (highest price first)
        self.asks = []  # Sell orders (lowest price first)

    def add_order(self, side, price, quantity, maker=None):
        order = {"side": side, "price": price, "quantity": quantity, "maker": maker}
        if side == "buy":
            self.bids.append(order)
            self.bids.sort(key=lambda x: -x["price"])
        else:
            self.asks.append(order)
            self.asks.sort(key=lambda x: x["price"])

    def match_order(self, side, quantity):
        trades = []
        book_side = self.asks if side == "buy" else self.bids
        while quantity > 0 and book_side:
            best_order = book_side[0]
            trade_qty = min(quantity, best_order["quantity"])
            trades.append({"price": best_order["price"], "quantity": trade_qty, "maker": best_order["maker"]})
            best_order["quantity"] -= trade_qty
            quantity -= trade_qty
            if best_order["quantity"] == 0:
                book_side.pop(0)
        return trades

    def get_best_bid(self):
        return self.bids[0] if self.bids else None

    def get_best_ask(self):
        return self.asks[0] if self.asks else None

    def get_depth(self, bin_size=1):
        from collections import defaultdict
        depth = defaultdict(lambda: {"buy": 0, "sell": 0})
        for order in self.bids:
            price_bin = bin_size * round(order["price"] / bin_size)
            depth[price_bin]["buy"] += order["quantity"]
        for order in self.asks:
            price_bin = bin_size * round(order["price"] / bin_size)
            depth[price_bin]["sell"] += order["quantity"]
        return depth

    def print_order_book(self):
        print("Order Book:")
        print("Bids:")
        for order in self.bids:
            print(f"Price: {order['price']}, Quantity: {order['quantity']}")
        print("Asks:")
        for order in self.asks:
            print(f"Price: {order['price']}, Quantity: {order['quantity']}")