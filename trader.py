import collections

import jsonpickle

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Any
import json
import numpy as np
from math import log, sqrt, exp, erf

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."


logger = Logger()


def get_price_info(order_depth: OrderDepth):
    sell_orders = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
    buy_orders = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

    # Orders are in form of Dict[price, amount]
    if list(buy_orders.items()) == []:
        return 0, 0, 0, 0, 0
    best_sell_price, _ = list(sell_orders.items())[0]
    best_buy_price, _ = list(buy_orders.items())[0]

    worst_sell_price, _ = list(sell_orders.items())[-1]
    worst_buy_price, _ = list(buy_orders.items())[-1]


    vol = 0
    tot = 0
    for ask_price, volume in sell_orders.items():
        tot += (ask_price * -volume)
        vol -= volume
    for bid_price, volume in buy_orders.items():
        tot += (bid_price * volume)
        vol += volume

    curr_mid_price = int(round(tot / vol))

    return curr_mid_price, best_buy_price, best_sell_price, worst_buy_price, worst_sell_price


def blackScholes(r, S, K, T, sigma):
    """Calculate BS price of call/put"""
    d1 = (log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    price = S * normal_cdf(d1) - K * exp(-r * T) * normal_cdf(d2)
    return int(round(price)), normal_cdf(d1)


def normal_cdf(x):
    """Cumulative distribution function for the standard normal distribution."""
    return (1 + erf(x / sqrt(2))) / 2

class Trader:

    def __init__(self):
        self.orchids_bid = None
        self.orchids_ask = None
        self.orchids_transport_fees = None
        self.orchids_export_tariff = None
        self.orchids_import_tariff = None
        self.orchids_sunlight = None
        self.orchids_humidity = None
        self.POSITION_LIMITS = {"AMETHYSTS": 20, "STARFRUIT": 20, "ORCHIDS": 100, "CHOCOLATE": 250, "STRAWBERRIES": 350,
                                "ROSES": 60, "GIFT_BASKET": 60, "COCONUT": 300, "COCONUT_COUPON": 600}
        self.position = {"AMETHYSTS": 0, "STARFRUIT": 0, "ORCHIDS": 0, "CHOCOLATE": 0, "STRAWBERRIES": 0, "ROSES": 0,
                         "GIFT_BASKET": 0, "COCONUT": 0, "COCONUT_COUPON": 0}
        self.starfruit_cache = []
        self.own_trades_orchids = []
        self.cache_limit = 17
        self.timestep = 0
        self.mav_basket_5 = []
        self.mav_basket_long = []
        self.macd_12_day = []
        self.macd_26_day = []

    def calc_next_price_starfruit(self, buy_orders, sell_orders):
        X = np.array([i for i in range(len(self.starfruit_cache))])
        Y = np.array(self.starfruit_cache)
        z = np.polyfit(X, Y, 1)
        p = np.poly1d(z)
        nxt_price = p(len(self.starfruit_cache))
        """intercept = 2.06636868

        coef = [0.33929074, 0.25998147, 0.21322955, 0.18708805]

        nxt_price = intercept
        for i, val in enumerate(self.starfruit_cache):
            nxt_price += val * coef[i]"""

        return int(round(nxt_price))

    def trade_starfruit_lr_last_few_timesteps(self, order_depth: OrderDepth, product: str):
        """
        We introduce a price lag aka we look at the last few timesteps and predict the next price from there.
        """

        orders: List[Order] = []

        sell_orders = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        buy_orders = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        position = self.position[product]  # Get the current position before the round.

        if len(self.starfruit_cache) == self.cache_limit:
            self.starfruit_cache.pop(0)

        curr_mid_price, best_buy_price, best_sell_price, _, _ = get_price_info(order_depth)
        self.starfruit_cache.append(curr_mid_price)

        if len(self.starfruit_cache) == self.cache_limit:
            next_price = self.calc_next_price_starfruit(buy_orders, sell_orders)
            bid = next_price - 2
            ask = next_price + 2

            for ask_price, volume in sell_orders.items():
                if ((ask_price <= bid) or (position < 0 and ask_price == bid+1)) and position < self.POSITION_LIMITS[product]:
                    max_amount_to_buy = self.POSITION_LIMITS[product] - position
                    order_for = min(-volume, max_amount_to_buy)
                    position += order_for
                    assert (order_for >= 0)
                    orders.append(Order(product, ask_price, order_for))

            undercut_buy = min(best_buy_price + 1, bid)

            if position < self.POSITION_LIMITS[product]:
                possible_volume = self.POSITION_LIMITS[product] - position
                orders.append(Order(product, undercut_buy, possible_volume))

            position = self.position[product]  # Reset position for selling

            # Same as above but for selling
            max_amount_to_sell = self.POSITION_LIMITS[product] + position
            for bid_price, volume in buy_orders.items():
                if ((bid_price >= ask) or (position > 5 and bid_price == ask-1)) and position > -self.POSITION_LIMITS[product]:
                    order_for = min(volume, max_amount_to_sell)
                    position -= order_for
                    assert (order_for >= 0)
                    orders.append(Order(product, bid_price, -order_for))

            # Now we are done selling on existing orders and create our own sell orders

            overcut_sell = max(best_sell_price - 1, ask)

            if position > -self.POSITION_LIMITS[product]:
                possible_volume = -self.POSITION_LIMITS[product] - position
                orders.append(Order(product, overcut_sell, possible_volume))

        return orders

    def trade_amethysts_best_bid_best_ask(self, order_depth: OrderDepth, product: str):
        bid = ask = 10000
        orders: List[Order] = []

        sell_orders = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        buy_orders = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        # Orders are in form of Dict[price, amount]
        best_sell_price, best_sell_amount = list(sell_orders.items())[0]
        best_buy_price, best_buy_amount = list(buy_orders.items())[0]

        position = self.position[product] # Get the current position before the round.

        # What we do here is we check if there is anything on the market that we can buy for less than bid price or
        # equal to bid if we have a negative position
        for ask_price, volume in sell_orders.items():
            if ((ask_price < bid) or (position < -5 and ask_price == bid)) and position < self.POSITION_LIMITS[product]:
                max_amount_to_buy = self.POSITION_LIMITS[product] - position
                order_for = min(-volume, max_amount_to_buy)
                position += order_for
                assert (order_for >= 0)
                orders.append(Order(product, ask_price, order_for))

        undercut_buy = best_buy_price + 1
        bid_pr = min(undercut_buy, bid - 1) # normal bid price

        if position < self.POSITION_LIMITS[product]:
            possible_volume = self.POSITION_LIMITS[product] - position
            orders.append(Order(product, bid_pr, possible_volume))
            position += possible_volume

        position = self.position[product]  # Reset position for selling

        # Same as above but for selling
        max_amount_to_sell = self.POSITION_LIMITS[product] + position
        for bid_price, volume in buy_orders.items():
            if ((bid_price > ask) or (position > 0 and bid_price == ask)) and position > -self.POSITION_LIMITS[product]:
                order_for = min(volume, max_amount_to_sell)
                position -= order_for
                assert (order_for >= 0)
                orders.append(Order(product, bid_price, -order_for))

        # Now we are done selling on existing orders and create our own sell orders

        overcut_sell = best_sell_price - 1
        sell_pr = max(overcut_sell, ask + 1) # We never want to sell for less than 10001

        if position > -self.POSITION_LIMITS[product]:
            possible_volume = -self.POSITION_LIMITS[product] - position
            orders.append(Order(product, sell_pr, possible_volume))

        return orders

    def update_orchids_information(self, orchids):
        self.orchids_bid = orchids.bidPrice
        self.orchids_ask = orchids.askPrice
        self.orchids_transport_fees = orchids.transportFees
        self.orchids_export_tariff = orchids.exportTariff
        self.orchids_import_tariff = orchids.importTariff
        self.orchids_sunlight = orchids.sunlight
        self.orchids_humidity = orchids.humidity

    def trade_orchids(self, order_depth: OrderDepth, product: str):
        orders: List[Order] = []
        # If you want to purchase 1 unit of ORCHID from the south, you will purchase at the askPrice,
        # pay the TRANSPORT_FEES, IMPORT_TARIFF
        purchase_price_from_south = self.orchids_ask + self.orchids_transport_fees + self.orchids_import_tariff

        # If you want to sell 1 unit of ORCHID to the south, you will sell at the bidPrice, pay the
        # TRANSPORT_FEES, EXPORT_TARIFF

        sell_price_to_south = self.orchids_bid - self.orchids_transport_fees - self.orchids_export_tariff

        # We also pay 0.1 seashells per timestamp that we are LONG on orchids

        our_mid_price, our_best_bid, our_best_ask, _, _ = get_price_info(order_depth)

        position = 0
        # Check our market to buy for less than what we can sell for in the south
        #logger.print(f"Best bid: {our_best_bid}, purchase price from south: {purchase_price_from_south}")
        #logger.print(f"Best ask: {our_best_ask}, sell price to south: {sell_price_to_south}")
        #logger.print(f"Our mid price: {our_mid_price}")

        diff_buy_from_south = our_mid_price - purchase_price_from_south
        diff_sell_to_south = sell_price_to_south - our_mid_price

        if diff_buy_from_south < diff_sell_to_south:
            for ask_price, volume in order_depth.sell_orders.items():
                if ask_price < sell_price_to_south and position < self.POSITION_LIMITS[product]:
                    max_amount_to_buy = self.POSITION_LIMITS[product] - position
                    order_for = min(-volume, max_amount_to_buy)
                    position += order_for
                    assert (order_for >= 0)
                    orders.append(Order(product, ask_price, order_for))
            if not (self.timestep % 1000000 == 0 and self.timestep != 0):
                if position < self.POSITION_LIMITS[product] and ((our_mid_price + 3) < sell_price_to_south):
                    possible_volume = self.POSITION_LIMITS[product] - position
                    try_buy_at_p_1 = int(4/6 * possible_volume)
                    try_buy_at_p_2 = int(2/6 * possible_volume)
                    try_buy_at_p_3 = possible_volume - try_buy_at_p_1 - try_buy_at_p_2
                    orders.append(Order(product, our_mid_price + 1, try_buy_at_p_1))
                    orders.append(Order(product, our_mid_price + 2, try_buy_at_p_2))
                    orders.append(Order(product, our_mid_price + 3, try_buy_at_p_3))
                    position += possible_volume
                elif position < self.POSITION_LIMITS[product] and ((our_mid_price + 2) < sell_price_to_south):
                    possible_volume = self.POSITION_LIMITS[product] - position
                    try_buy_at_p_1 = int(5/6 * possible_volume)
                    try_buy_at_p_2 = possible_volume - try_buy_at_p_1
                    orders.append(Order(product, our_mid_price + 1, try_buy_at_p_1))
                    orders.append(Order(product, our_mid_price + 2, try_buy_at_p_2))
                    position += possible_volume
                elif position < self.POSITION_LIMITS[product] and ((our_mid_price + 1) < sell_price_to_south):
                    possible_volume = self.POSITION_LIMITS[product] - position
                    orders.append(Order(product, our_mid_price + 1, possible_volume))
                    position += possible_volume
        else:
            # Check our market to sell for more than what we can buy for in the south
            for bid_price, volume in order_depth.buy_orders.items():
                if bid_price > purchase_price_from_south and position > -self.POSITION_LIMITS[product]:
                    max_amount_to_sell = self.POSITION_LIMITS[product] + position
                    order_for = min(volume, max_amount_to_sell)
                    position -= order_for
                    #logger.print(f"I see that the bid price {bid_price} is more than the purchase price from the south {purchase_price_from_south} and I can sell {order_for} units")
                    assert (order_for >= 0)
                    orders.append(Order(product, bid_price, -order_for))
            if not (self.timestep % 1000000 == 0 and self.timestep != 0):
                if position > -self.POSITION_LIMITS[product] and ((our_mid_price - 3) > purchase_price_from_south):
                    possible_volume = -self.POSITION_LIMITS[product] - position
                    try_sell_at_p_1 = int(7/12 * possible_volume)
                    try_sell_at_p_2 = int(5/12 * possible_volume)
                    try_sell_at_p_3 = possible_volume - try_sell_at_p_1 - try_sell_at_p_2
                    orders.append(Order(product, our_mid_price - 1, try_sell_at_p_1))
                    orders.append(Order(product, our_mid_price - 2, try_sell_at_p_2))
                    orders.append(Order(product, (our_mid_price - 3), try_sell_at_p_3))
                elif position > -self.POSITION_LIMITS[product] and ((our_mid_price - 2) > purchase_price_from_south):
                    possible_volume = -self.POSITION_LIMITS[product] - position
                    try_sell_at_p_1 = int(9/12 * possible_volume)
                    try_sell_at_p_2 = possible_volume - try_sell_at_p_1
                    orders.append(Order(product, our_mid_price - 1, try_sell_at_p_1))
                    orders.append(Order(product, our_mid_price - 2, try_sell_at_p_2))
                # TODO: Maybe less volume here?
                elif position > -self.POSITION_LIMITS[product] and ((our_mid_price - 1) > purchase_price_from_south):
                    possible_volume = -self.POSITION_LIMITS[product] - position
                    orders.append(Order(product, our_mid_price - 1, possible_volume))
                elif position > -self.POSITION_LIMITS[product] and (our_mid_price > purchase_price_from_south):
                    possible_volume = -self.POSITION_LIMITS[product] - position
                    orders.append(Order(product, our_mid_price, possible_volume))
        #logger.print(f"My position is {position} and my conversion is {-self.position[product]}")

        conversions = -self.position[product]

        return orders, conversions

    def trade_baskets(self, order_depths: dict[str, OrderDepth]) -> dict[str, list[Order]]:
        products = ["CHOCOLATE", "STRAWBERRIES", "ROSES", "GIFT_BASKET"]
        orders = {"CHOCOLATE": [], "STRAWBERRIES": [], "ROSES": [], "GIFT_BASKET": []}

        best_sell, best_buy, worst_sell, worst_buy, mid_price, = {}, {}, {}, {}, {}

        for product in products:
            order_depth: OrderDepth = order_depths[product]

            p_mid_price, p_best_buy, p_best_sell, p_worst_buy, p_worst_sell = get_price_info(order_depth)
            mid_price[product] = p_mid_price
            best_buy[product] = p_best_buy
            best_sell[product] = p_best_sell
            worst_buy[product] = p_worst_buy
            worst_sell[product] = p_worst_sell

        if len(self.mav_basket_5) == 5:
            self.mav_basket_5.pop(0)

        if len(self.mav_basket_long) == 20:
            self.mav_basket_long.pop(0)

        price_combined = 4 * mid_price["CHOCOLATE"] + 6* mid_price["STRAWBERRIES"] +  mid_price["ROSES"]

        self.mav_basket_5.append(mid_price["GIFT_BASKET"])
        self.mav_basket_long.append(mid_price["GIFT_BASKET"])

        if len(self.mav_basket_long) == 20:
            mav5 = np.array(self.mav_basket_5).mean()
            mavlong = np.array(self.mav_basket_long).mean()
            mavlong_std = np.array(self.mav_basket_long).std()
            zscore = (mav5 - mavlong) / mavlong_std
            logger.print(f"Z-score: {zscore}")
            logger.print(f"difference in mav: {mav5 - mavlong}")
            logger.print(f"and price basket: {mid_price['GIFT_BASKET']}")
            logger.print(f"and price combined: {price_combined}")
            logger.print(f" DIfference: {mid_price['GIFT_BASKET'] - price_combined}")
            if zscore < - 1:
                volume = self.POSITION_LIMITS["GIFT_BASKET"] - self.position["GIFT_BASKET"]
                # the basket is undervalued so we can buy it
                logger.print(f"The basket is undervalued so we can buy it")
                orders["GIFT_BASKET"].append(Order("GIFT_BASKET", best_sell["GIFT_BASKET"], volume))
            elif zscore > 1:
                volume = self.POSITION_LIMITS["GIFT_BASKET"] + self.position["GIFT_BASKET"]
                logger.print(f"The basket is overvalued so we can sell it")
                # the basket is overvalued so we can sell it
                orders["GIFT_BASKET"].append(Order("GIFT_BASKET", best_buy["GIFT_BASKET"], -volume))
            elif 0.3 > zscore > -0.3:
                if self.position["GIFT_BASKET"] > 0:
                    volume = self.position["GIFT_BASKET"]
                    orders["GIFT_BASKET"].append(Order("GIFT_BASKET", best_sell["GIFT_BASKET"], -volume))
                elif self.position["GIFT_BASKET"] < 0:
                    volume = -self.position["GIFT_BASKET"]
                    orders["GIFT_BASKET"].append(Order("GIFT_BASKET", best_buy["GIFT_BASKET"], volume))

        return orders

    def trade_options(self, order_depths: dict[str, OrderDepth]) -> dict[str, list[Order]]:
        products = ["COCONUT", "COCONUT_COUPON"]
        orders = {"COCONUT": [], "COCONUT_COUPON": []}


        best_sell, best_buy, worst_sell, worst_buy, mid_price, = {}, {}, {}, {}, {}

        for product in products:
            order_depth: OrderDepth = order_depths[product]

            p_mid_price, p_best_buy, p_best_sell, p_worst_buy, p_worst_sell = get_price_info(order_depth)
            mid_price[product] = p_mid_price
            best_buy[product] = p_best_buy
            best_sell[product] = p_best_sell
            worst_buy[product] = p_worst_buy
            worst_sell[product] = p_worst_sell
        T = 247/252
        r = 0
        K = 10000
        sigma = 0.1609616171503603
        S = mid_price["COCONUT"]


        fair_price_option, delta = blackScholes(r, S, K, T, sigma)
        logger.print(f"Fair price: {fair_price_option} and current market price: {mid_price['COCONUT_COUPON']}\n"
                     f"Difference: {fair_price_option - mid_price['COCONUT_COUPON']}")

        diff = fair_price_option - mid_price["COCONUT_COUPON"]
        #  delta = 0.5
        # coco coupon: 600
        # total_delta/desired_pos = 300
        # desired_pos = total_delta

        desired_pos = -int(delta * self.position["COCONUT_COUPON"]) # - 300 300

        current_pos = self.position["COCONUT"] # 0

        amount_to_handle = abs(current_pos - desired_pos) # 0 - (-300) = 300

        # check if i should buy or sell coconuts

        #num_orders = min(-vol, limit - position)

        #desired_pos = int(total_delta + self.position["COCONUT"]) #


        logger.print(f"Delta: {delta} \nOptionPos:{self.position['COCONUT_COUPON']} \nStockPos:{self.position['COCONUT']}")
        logger.print(f"Total delta:  and desired position: {desired_pos}")
        
        if desired_pos < current_pos:

            orders["COCONUT"].append(Order("COCONUT", best_buy["COCONUT"], -amount_to_handle))
        elif desired_pos > current_pos:
            # sell coconuts
            orders["COCONUT"].append(Order("COCONUT", best_sell["COCONUT"], amount_to_handle))

        fair_bid = min(fair_price_option - 5, best_buy["COCONUT_COUPON"]+1)
        fair_ask = max(fair_price_option + 5, best_sell["COCONUT_COUPON"]-1)

        logger.print(f"According to Blackscholes, fair price is {fair_price_option} and we have a mid price of {mid_price['COCONUT_COUPON']}")
        if mid_price["COCONUT_COUPON"] < fair_price_option:
            logger.print(f"Option is undervalued and it is a good idea to buy it")
        else:
            logger.print(f"Option is overvalued and it is a good idea to sell it")
        position = self.position["COCONUT_COUPON"]

        sell_orders = collections.OrderedDict(sorted(order_depths["COCONUT_COUPON"].sell_orders.items()))
        buy_orders = collections.OrderedDict(sorted(order_depths["COCONUT_COUPON"].buy_orders.items(), reverse=True))

        # What we do here is we check if there is anything on the market that we can buy for less than bid price or
        # equal to bid if we have a negative position
        """for ask_price, volume in sell_orders.items():
            if (ask_price <= fair_bid) and position < self.POSITION_LIMITS["COCONUT_COUPON"]: #or (position < -5 and ask_price == bid))
                max_amount_to_buy = self.POSITION_LIMITS["COCONUT_COUPON"] - position
                order_for = min(-volume, max_amount_to_buy)
                position += order_for
                assert (order_for >= 0)
                orders["COCONUT_COUPON"].append(Order("COCONUT_COUPON", ask_price, order_for))"""


        if diff > 5:
            if position < self.POSITION_LIMITS["COCONUT_COUPON"]:
                possible_volume = self.POSITION_LIMITS["COCONUT_COUPON"] - position
                orders["COCONUT_COUPON"].append(Order("COCONUT_COUPON", fair_bid, possible_volume))
                position += possible_volume

        position = self.position["COCONUT_COUPON"]  # Reset position for selling

        # Same as above but for selling
        max_amount_to_sell = self.POSITION_LIMITS["COCONUT_COUPON"] + position
        """for bid_price, volume in buy_orders.items():
            if (bid_price >= fair_ask) and position > -self.POSITION_LIMITS["COCONUT_COUPON"]:
                order_for = min(volume, max_amount_to_sell)
                position -= order_for
                assert (order_for >= 0)
                orders["COCONUT_COUPON"].append(Order("COCONUT_COUPON", bid_price, -order_for))"""


        if diff < -5:
            if position > -self.POSITION_LIMITS["COCONUT_COUPON"]:
                possible_volume = -self.POSITION_LIMITS["COCONUT_COUPON"] - position
                orders["COCONUT_COUPON"].append(Order("COCONUT_COUPON", fair_ask, possible_volume))

        return orders




    def cache_results(self):
        return {
            "starfruit_cache": self.starfruit_cache,
            "mav_basket_5": self.mav_basket_5,
            "mav_basket_long": self.mav_basket_long,
            "macd_12_day": self.macd_12_day,
            "macd_26_day": self.macd_26_day,
        }

    def uncache_results(self, trader_data):
        if trader_data:
            data = jsonpickle.decode(trader_data)
            self.starfruit_cache = data["starfruit_cache"]
            self.mav_basket_5 = data["mav_basket_5"]
            self.mav_basket_long = data["mav_basket_long"]
            self.macd_12_day = data["macd_12_day"]
            self.macd_26_day = data["macd_26_day"]

    def calculate_orders(self, product, order_depth, our_bid, our_ask, orchild=False, gift_basket=False):
        orders: list[Order] = []

        sell_orders = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        buy_orders = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_price = self.get_volume_and_best_price(sell_orders, buy_order=False)
        buy_vol, best_buy_price = self.get_volume_and_best_price(buy_orders, buy_order=True)

        logger.print(f'Product: {product} - best sell: {best_sell_price}, best buy: {best_buy_price}')

        position = self.position[product] if not orchild else 0
        limit = self.POSITION_LIMITS[product]

        # penny the current highest bid / lowest ask
        penny_buy = best_buy_price + 1
        penny_sell = best_sell_price - 1

        bid_price = our_bid
        ask_price = our_ask

        # MARKET TAKE ASKS (buy items)
        for ask, vol in sell_orders.items():
            if position < limit and (ask <= our_bid or (position < 0 and ask == our_bid + 1)):
                num_orders = min(-vol, limit - position)
                position += num_orders
                orders.append(Order(product, ask, num_orders))

        # MARKET MAKE BY PENNYING
        if position < limit:
            num_orders = limit - position
            orders.append(Order(product, bid_price, num_orders))
            position += num_orders

        # RESET POSITION
        position = self.position[product] if not orchild else 0

        # MARKET TAKE BIDS (sell items)
        for bid, vol in buy_orders.items():
            if position > -limit and (bid >= our_ask or (position > 0 and bid + 1 == our_ask)):
                num_orders = max(-vol, -limit - position)
                position += num_orders
                orders.append(Order(product, bid, num_orders))

        # MARKET MAKE BY PENNYING
        if position > -limit:
            num_orders = -limit - position
            orders.append(Order(product, ask_price, num_orders))
            position += num_orders

        return orders
    def get_volume_and_best_price(self, orders, buy_order):
        volume = 0
        best = 0 if buy_order else int(1e9)

        for price, vol in orders.items():
            if buy_order:
                volume += vol
                best = max(best, price)
            else:
                volume -= vol
                best = min(best, price)

        return volume, best

    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        logger.print("traderData: " + state.traderData)
        logger.print("Observations: " + str(state.observations))
        conversion_observations = state.observations.conversionObservations
        # The values from the south island
        self.own_trades_orchids = state.own_trades.get("ORCHIDS", [])
        result = {}
        conversions = -1
        self.uncache_results(state.traderData)
        self.timestep = state.timestamp
        for product in self.position:
            position: int = state.position.get(product, 0)  # Because if we don't have a position, it throws error
            self.position[product] = position  # Add this globally so we can access it
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            if product == "AMETHYSTS":
                orders = self.trade_amethysts_best_bid_best_ask(order_depth, product)
                result[product] = orders
            elif product == "STARFRUIT":
                orders = self.trade_starfruit_lr_last_few_timesteps(order_depth, product)
                result[product] = orders
            elif product == "ORCHIDS":
                self.update_orchids_information(conversion_observations["ORCHIDS"])
                orders, conversions = self.trade_orchids(order_depth, product)
                result[product] = orders
            elif product == 'GIFT_BASKET':
                DIFFERENCE_MEAN = 383
                DIFFERENCE_STD = 76.42438217375009
                PERCENT_OF_STD_TO_TRADE_AT = 0.72
                basket_items = ['GIFT_BASKET', 'CHOCOLATE', 'STRAWBERRIES', 'ROSES']
                mid_price = {}

                for item in basket_items:
                    _, best_sell_price = self.get_volume_and_best_price(state.order_depths[item].sell_orders,
                                                                        buy_order=False)
                    _, best_buy_price = self.get_volume_and_best_price(state.order_depths[item].buy_orders,
                                                                       buy_order=True)

                    mid_price[item] = (best_sell_price + best_buy_price) / 2

                difference = mid_price['GIFT_BASKET'] - 4 * mid_price['CHOCOLATE'] - 6 * mid_price['STRAWBERRIES'] - \
                             mid_price['ROSES'] - DIFFERENCE_MEAN

                worst_bid_price = min(order_depth.buy_orders.keys())
                worst_ask_price = max(order_depth.sell_orders.keys())

                if difference > PERCENT_OF_STD_TO_TRADE_AT * DIFFERENCE_STD:  # basket overvalued, sell
                    orders = self.calculate_orders(product, order_depth, -int(1e9), worst_bid_price, gift_basket=True)
                    result[product] = orders

                elif difference < -PERCENT_OF_STD_TO_TRADE_AT * DIFFERENCE_STD:  # basket undervalued, buy
                    orders = self.calculate_orders(product, order_depth, worst_ask_price, int(1e9), gift_basket=True)
                    result[product] = orders


        option_orders = self.trade_options(state.order_depths)
        for product, orders in option_orders.items():
            result[product] = orders

        dict = self.cache_results()
        traderData = jsonpickle.encode(dict)  # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.

        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData


