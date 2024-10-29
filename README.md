# IMC Prosperity 2

<img src= "https://github.com/user-attachments/assets/ebe1eefd-7670-4676-b677-d0ffbef7fd01" width="500"> <img src=  "https://github.com/user-attachments/assets/d16151d5-8c5d-46db-a0a0-1556235b3c30" width="500">

[IMC Prosperity 2](https://prosperity.imc.com/) Trade Challenge, hosted by [IMC Trading](https://www.imc.com/eu), was the second installment of a global algorithmic trading competition with over 9000 teams worldwide. We achieved a global ranking of 29th and a national (UK) ranking of 2nd. 

Over the 15(+2) day challenge we were met with a variety of manual and algorithmic trading challenges. This was a very fun and interesting competition, below are the strategies we used for each round:
- R1: Market Making
- R2: Arbitrage
- R3: ETF Arb/Pairs Trading
- R4: Options Trading
- R5: Signals


## Round 1
The algo forecasts short-term price changes for starfruit using a predictive trading method that employs linear regression on previous prices. In order to profit from expected market moves, it bases its bid and ask orders on this projection, setting them just below the anticipated price. For amethysts, the algo uses a market-making approach, aiming to make money off of the bid-ask spread. It places orders at slightly modified bid and ask prices, undercutting current orders. 

## Round 2
The approach for orchids incorporates dynamic bid-ask changes, external market forces, and arbitrage. The algo accounts for external fees, customs, and freight expenses while determining possible purchase and sell prices. With the goal of purchasing when market circumstances suggest a profit from a future sale and vice versa, orders are strategically set based on pricing discrepancies and evaluates the profit potential between the local and exterior markets. By gradually entering or leaving the market in reaction to advantageous pricing, it also uses the mid-price spread to divide volume into smaller, spaced orders, balancing risk and profitability.

## Round 3
For chocolate, strawberries, roses, and gift baskets, the strategy is based of ETF Arbitrage and relies on a mean reversion approach by analyzing the combined gift basket. The algo tracks short- and long-term moving averages (5 and 20 periods) to calculate a Z-score that identifies when the gift basket's market price deviates from its expected value, as derived from individual component prices. When the gift basket is undervalued (low Z-score), the bot buys it, anticipating price correction. Conversely, it sells the basket when overvalued (high Z-score). If the Z-score is near zero, it adjusts its position back to neutral to minimize holding costs, with the aim of capitalizing on temporary market inefficiencies between the basket and its components.

## Round 4
For coconuts and coconut coupons, the strategy follows option pricing and utilizes the Black-Scholes model to calculate the theoretical fair price and delta of the coconut coupon. The algo assesses market prices against this fair price, identifying whether the coupon is undervalued or overvalued. If undervalued (market price < fair price), it buys coconut coupons, while if overvalued (market price > fair price), it sells them. Additionally, delta hedging is utilised by adjusting its coconut position to match the coupon’s calculated delta, mitigating risk from price changes in coconuts. This approach aims to profit from mispricings while maintaining a balanced exposure.

## Round 5
This round comprised of data relating to trades that bots had made throughout the competition. The goal was to obtain viable signals on the bots in order to gain more profit. EDA suggested that shorting Roses if Rhianna bought roses could be a viable signal but this was not implemeneted due to the uncertainty around how strong the signla was. So the eventual decision was to not follow any signals and optimise the prior strategies in previous rounds
