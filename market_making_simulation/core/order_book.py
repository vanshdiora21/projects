
import time

class OrderBook:
    def __init__(self):
        self.bids = []  # List of dicts: {"price": ..., "quantity": ..., "timestamp": ...}
        self.asks = []

    def add_order(self, side, price, quantity, maker="MM1"):
        order = {"side": side, "price": price, "quantity": quantity, "maker": maker}
        
        if side == "buy":
            self.bids.append(order)
            self.bids = sorted(self.bids, key=lambda x: -x["price"])  # Highest price first
        else:
            self.asks.append(order)
            self.asks = sorted(self.asks, key=lambda x: x["price"])  # Lowest price first


    def match_order(self, order_type, quantity):
        trades = []
        if order_type == "buy":
            while quantity > 0 and self.asks:
                best_ask = self.asks[0]
                trade_qty = min(quantity, best_ask["quantity"])
                trades.append({"price": best_ask["price"], "quantity": trade_qty})
                quantity -= trade_qty
                best_ask["quantity"] -= trade_qty
                if best_ask["quantity"] == 0:
                    self.asks.pop(0)
        elif order_type == "sell":
            while quantity > 0 and self.bids:
                best_bid = self.bids[0]
                trade_qty = min(quantity, best_bid["quantity"])
                trades.append({"price": best_bid["price"], "quantity": trade_qty})
                quantity -= trade_qty
                best_bid["quantity"] -= trade_qty
                if best_bid["quantity"] == 0:
                    self.bids.pop(0)
        return trades

    def get_best_bid(self):
        return self.bids[0] if self.bids else None

    def get_best_ask(self):
        return self.asks[0] if self.asks else None

    def print_order_book(self):
        print("Order Book:")
        print("Bids:")
        for bid in self.bids:
            print(f"Price: {bid['price']}, Quantity: {bid['quantity']}")
        print("Asks:")
        for ask in self.asks:
            print(f"Price: {ask['price']}, Quantity: {ask['quantity']}")
    def get_depth(self, bin_size=1):
        """
        Returns order book depth aggregated by price bins.
        """
        from collections import defaultdict
        depth = defaultdict(lambda: {"buy": 0, "sell": 0})

        # Aggregate bids
        for order in self.bids:
            price_bin = bin_size * round(order["price"] / bin_size)
            depth[price_bin]["buy"] += order["quantity"]

        # Aggregate asks
        for order in self.asks:
            price_bin = bin_size * round(order["price"] / bin_size)
            depth[price_bin]["sell"] += order["quantity"]

        return depth
