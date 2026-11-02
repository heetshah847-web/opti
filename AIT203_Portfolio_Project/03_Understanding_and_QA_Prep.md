# Understanding & Q&A Prep

Read this to understand the ideas behind the slides, so you can explain them in your own words and handle questions.

---

## The core idea in one paragraph

We are **not** predicting future stock prices. We're choosing the best way to split our money across 18 chosen stocks. From five years of historical prices, we estimate each stock's average return and how every pair of stocks moves together. We measure the portfolio's risk as the variance of its returns, which depends on both. Then we find the split with the lowest risk that still meets our target return, using the whole budget with no negative positions. Changing the target return shows the full trade-off between risk and return.

**One honest assumption:** we use the past 5 years to estimate how these stocks behave. That's standard practice, but the future can differ.

---

## Key terms

- **Adjusted close:** the daily closing price, corrected for stock splits and dividends, so every jump in the data is a real gain or loss.
- **Dividend:** cash a company pays its shareholders out of its profits. The share price drops by roughly that amount on the payout date, but the investor hasn't lost anything: they received it as cash.
- **Stock split:** one share becomes several cheaper shares. The price drops, but the total value doesn't change.
- **Daily return:** (today - yesterday) / yesterday. We use returns, not prices, because returns put every stock on the same scale.
- **Expected return (μ):** each stock's average daily return, used as an estimate of what it tends to earn.
- **Covariance matrix (Σ):** an 18 × 18 table showing how every pair of stocks moves together. It contains 18 "own risk" values plus 153 pair values.
- **Risk (variance):** how much the portfolio's return swings around its average. Its square root is volatility. The risk of a split depends on each stock's own swings and on how every pair moves together.
- **r_target:** the minimum return we insist on. Raising it forces the optimizer toward bumpier stocks, so risk goes up. Lowering it lets risk go down.
- **Efficient frontier:** the best possible risk for every level of return.
- **Short selling:** borrowing a stock and selling it to bet it will fall. We don't allow it.

---

## Slide 6: the six reasons, explained simply

1. **There's no answer key.** "Will it rain tomorrow?" gets checked tomorrow. "Which route should I take to college?" never does: you'll never know if Route B was better. Stock prices are like the rain question; splitting money is like the route question. So we define "best" ourselves: lowest risk for our target return.
2. **Two goals pull against each other.** Nvidia and Tesla can grow a lot but swing wildly; Coca-Cola is calm but grows slowly. Chasing only return means putting everything in one wild stock. Avoiding only risk means earning almost nothing. So we fix the return and minimize risk.
3. **Risk doesn't simply add up.** Own an ice-cream shop and an umbrella shop, and you earn every day. Own two ice-cream shops, and both lose on rainy days. Exxon and Chevron are like two ice-cream shops. In 2022, energy stocks rose while big tech fell, so a mix of both was much calmer. With 18 stocks there are 153 pairs to track.
4. **Every choice moves every other.** Like slices of one pizza: a bigger slice for one person means a smaller slice for someone else. You can't tune 18 numbers one at a time.
5. **All the rules, all at once.** Move money into Coca-Cola to cut risk, and the return drops below target. Add Nvidia to fix the return, and risk rises again. By hand, you go in circles.
6. **Guessing can't prove anything.** Weights can be any fraction, so there's no list to check. Walking a foggy valley, you find a low spot but can't know if there's a lower one. Optimization is the map: it finds the lowest point and confirms nothing is lower, under our rules.

---

## Slide 7: the impact, explained simply

- **Gut feel:** people lean toward familiar names (Apple, Microsoft, Google, Amazon, Meta, Nvidia), which are mostly tech-related and often move together. The portfolio feels diversified but hides concentrated risk.
- **Equal split:** about 5.6% in each stock. It gives jumpy Tesla the same share as calm Coca-Cola and ignores how stocks balance each other.
- **The strongest point:** the equal split follows all our rules, so it's one of the options the optimizer considers. The optimized split can **never be worse** than it on our data.
- **With optimization:** the same 18 stocks, weighted so their ups and downs offset each other. The same return with less risk, or more return for the same risk.
- **Who it matters for:** anyone investing a fixed amount, from a student's first savings to a large fund. Less risk means smaller drops when markets fall.
- **Price of realism:** real investors can't hold every stock. Limit how many we can own and the best result can only stay the same or get slightly worse. That gap is the price of realism (the professor's own term from the project PDF).

---

## Likely questions and answers

**"Isn't this just machine learning / prediction?"**
No. We use historical averages as inputs, but our job is choosing the split. Nothing ever "checks" a split against reality the way a price forecast gets checked.

**"Couldn't you just look back and say the best option was the stock with the highest return?"**
Only in hindsight. In January 2021 nobody knew which stock would win. Putting everything in one stock is a gamble: Nvidia itself lost around half its value during 2022. And the "winner" changes depending on which years you look at. The future shows you how a choice turned out, not whether it was the right choice.

**"Why not just maximize return?"**
The answer would be 100% in the single highest-return stock, ignoring risk completely.

**"How do you know optimization beats an equal split?"**
The equal split obeys all our rules, so it's one of the options the optimizer considers. The optimum can never be worse than it on our data.

**"Will this actually do better in the future?"**
Not guaranteed. Our numbers come from past data, and the future can behave differently. Testing how well it holds up is part of our later analysis.

**"Why 18 stocks, and why these?"**
Large, well-known companies with complete price histories, spread across 7 sectors so they genuinely behave differently. It's one fixed list: every number we compute is specific to these 18.

**"Aren't the stocks your data points, not your features?"**
We treat each trading day as one observation of all 18 stocks at once. That's how the covariance matrix gets estimated.

**"What about inflation?"**
We use nominal historical returns. Inflation's effects, including how it hits sectors differently, are already reflected in each stock's historical returns and covariance.

**"Why limit the number of stocks later?"**
Fewer positions are easier to manage and cheaper to trade, and it lets us measure the cost of being realistic.

---

## The 18 stocks by GICS sector

| Sector | Stocks |
|---|---|
| Information Technology | AAPL (Apple), MSFT (Microsoft), NVDA (Nvidia) |
| Communication Services | GOOGL (Alphabet), META (Meta), DIS (Disney) |
| Consumer Discretionary | AMZN (Amazon), TSLA (Tesla), HD (Home Depot) |
| Consumer Staples | PG (Procter & Gamble), KO (Coca-Cola), PEP (PepsiCo) |
| Financials | JPM (JPMorgan Chase), V (Visa) |
| Energy | XOM (Exxon Mobil), CVX (Chevron) |
| Health Care | JNJ (Johnson & Johnson), UNH (UnitedHealth Group) |

Note: Amazon, Tesla, Alphabet, Meta and Disney are often called "tech" but are not in the GICS Information Technology sector. Visa moved from Information Technology to Financials in a 2023 GICS revision.
