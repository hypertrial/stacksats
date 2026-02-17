---
title: Glossary
description: Definitions of key Bitcoin and StackSats system terms.
---

# Glossary

This glossary defines common terms used throughout the StackSats documentation, covering both Bitcoin-specific concepts and system-specific mechanics.

## Bitcoin Terms

### Sats (Satoshis)
The smallest unit of Bitcoin.
- `1 Bitcoin (BTC) = 100,000,000 sats`
- `1 sat = 0.00000001 BTC`

Strategies in this framework focus on maximizing the number of sats accumulated per dollar invested.

### MVRV (Market Value to Realized Value)
A ratio used to assess whether Bitcoin is overvalued or undervalued relative to its "fair value".
- **Market Value:** Current price $\times$ circulating supply.
- **Realized Value:** The sum of the value of all coins at the price they last moved on-chain. This is often considered the aggregate "cost basis" of the network.
- **Interpretation:**
    - High MVRV (> 3.0) often indicates market tops (price is far above cost basis).
    - Low MVRV (< 1.0) often indicates market bottoms (price is below cost basis).

### Halving
An event that occurs approximately every 4 years where the rate of new Bitcoin issuance (block subsidy) is cut in half. This is a key component of Bitcoin's disinflationary monetary policy and often delineates 4-year market cycles.

### Block Subsidy
The amount of new Bitcoin created in each block (approximately every 10 minutes) and awarded to the miner. This subsidy halves every 210,000 blocks.

---

## System Terms

### Allocation Span
The fixed window of time (default: 365 days) over which a fixed budget must be fully allocated. The model solves for the optimal distribution of capital over this rolling window.

### Feasibility Clipping
The mechanism by which the framework ensures that a proposed daily weight is valid. It checks:
1.  **Non-negativity:** Weights cannot be negative.
2.  **Maximum Weight:** Access to daily liquidity is capped (default: 10% of total budget).
3.  **Remaining Budget:** The weight cannot exceed the budget remaining for the current allocation span.

### Locked Prefix
The portion of the allocation timeline that is in the past. Once a day passes, its allocation is "locked" and cannot be changed by the strategy. This immutability ensures realistic backtesting and prevents "repainting" of history.

### Sats per Dollar (SPD)
The primary efficiency metric for accumulation strategies.
- **Formula:** $\text{Total Sats Accumulated} / \text{Total USD Invested}$
- **Meaning:** It represents the weighted average purchasing power of your capital. A higher SPD means you acquired more Bitcoin for the same amount of dollars compared to a benchmark (like uniform DCA).
- **Why it matters:** In an accumulation strategy, the goal is to lower your average cost basis. Maximizing SPD is mathematically equivalent to minimizing average cost basis.
