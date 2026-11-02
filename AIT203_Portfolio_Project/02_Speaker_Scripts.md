# Speaker Scripts: Proposal & Motivation (5 minutes total)

**Project:** Optimal Portfolio Allocation Across a Diversified Stock Portfolio
**Course:** AIT 203, Section A
**Team:** Yashas Yogindra, Heet Shah, Pushya Jain

| Speaker | Slides | Time | Rubric items covered |
|---|---|---|---|
| Pushya | 1 to 2 | ~1 min | Group members, chosen data, the problem in plain words |
| Yashas | 2 (dark box) to 5 | ~2 min | Source, size (data points, features), what the key variables represent |
| Heet | 6 to 7 | ~2 min | Why we need optimization, impact of solving it |

Do one timed run-through together. If you go over 5 minutes, cut the "Yahoo also gives open, high, low..." sentence (Yashas) and merge reasons four and five (Heet).

---

## Pushya (about 1 minute)

**[Slide 1: Title]**
Good morning, everyone. I'm Pushya, and with me are Yashas and Heet. Our project is called Optimal Portfolio Allocation. It's from Option 6 on the project list, Finance and Economics, and all our data comes from Yahoo Finance, which is free and public.

**[Slide 2: The Problem]**
So, what's the problem? Imagine you have some money to invest, and you've already chosen 18 well-known companies, like Apple, Coca-Cola and Disney. The question is: how much of your money should go into each one?

Most people would just split it evenly, or go with their gut. But two splits that look almost the same can carry very different amounts of risk.

So our goal is to find the safest split that still earns the amount we're aiming for. We're not trying to predict stock prices. We're deciding how to divide money we already have.

Over to Yashas.

---

## Yashas (about 2 minutes)

**[Slide 2: point to the dark box]**
Thanks, Pushya. The box on this slide is that same question, written in maths. We choose 18 percentages, one for each stock. We want the overall risk to be as small as possible, under three rules: we invest the whole budget, no stock gets a negative amount, and we have to reach our target return. That's all the maths we need today.

**[Slide 3: Dataset & Source]**
Our data is Yahoo Finance's historical daily prices, from 2021 to 2026. That gives us about 1,260 data points, one for each trading day, and 18 features, one for each stock. That's roughly 22,700 prices in total.

**[Slide 4: Key Variables]**
Here's what that data looks like. Each row is a trading day, so one row is a snapshot of all 18 stocks on that day. Each column is one stock. And every cell is an adjusted closing price. "Adjusted" means it's corrected for things like stock splits and dividends, so every jump in the data is a real gain or loss. Yahoo also gives open, high, low, volume and the unadjusted close, but we only need the adjusted close.

From those prices we work out three things: the daily return, which is the percentage change from one day to the next; each stock's expected return, which we estimate as its average daily return; and the covariance matrix, which shows how every pair of stocks moves together. That last one is how we measure risk.

**[Slide 5: Our 18 Stocks]**
And these are our 18 stocks. It's one fixed list, not any random 18, so everything we calculate is specific to these companies. They come from seven different sectors, from technology and banking to energy and health care, because stocks from different sectors behave differently, and that gives us something real to work with.

Heet will now explain why this actually needs optimization.

---

## Heet (about 2 minutes)

**[Slide 6: Why This Needs Optimization]**
Thanks, Yashas. So why can't we just pick a split ourselves? There are six reasons.

One: there's no answer key. The future shows you how a split turned out, but not whether it was the right choice. So we have to define what "best" means ourselves, and then find it.

Two: our goals pull against each other. More return usually means more risk, so we fix the return we want and make the risk as small as possible.

Three: risk doesn't simply add up. Stocks that rise and fall together, like two oil companies, don't make you much safer. Stocks that move independently partly cancel each other out. Across 18 stocks, that's 153 pairs to keep track of.

Four: the percentages have to add up to 100, so changing one changes the others.

Five: every rule has to hold at the same time. If you adjust by hand, fixing one rule tends to break another.

And six: the percentages can be any fraction, so there's no list you can check. A guess might look decent, but it can never prove it's the best. Optimization can.

**[Slide 7: The Impact]**
So what's the impact? Most people split money by gut feel: evenly, or more into names they recognize. Neither looks at how the stocks move together, and that's exactly where extra risk hides.

An even split carries more risk than it needs to for the return it earns. Right now that's a claim, and solving this turns it into a measured number: the same return with less risk, or more return for the same risk.

That matters for anyone investing a fixed amount, from a student's first savings to a large fund. Less risk means smaller swings when markets fall.

And since real investors can't hold every stock, we'll also measure how much the best result drops when we limit how many stocks we're allowed to own. That gap is called the price of realism.

Thank you. We're happy to take any questions.
