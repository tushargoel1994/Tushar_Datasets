+**What this dataset really is (the honest reading):**

It's a transactional fact table: 16 columns covering Segment (5), Country (4), Product (2 visible: Carretera, Montana — full set typically has 6), Discount Band, Units, Manufacturing Price, Sale Price, Gross Sales, Discounts, Net Sales, COGS, Profit, and a date dimension.

**The non-obvious patterns I noticed (these are the strategic hooks):**

1. **The Sale Price column is bizarre.** The same product (Carretera) sells at $20, $15, and $350 to different segments. Montana sells at $7, $12, $15, $20, $125, $300. Manufacturing cost is fixed at $3 and $5. This isn't pricing — it's a *segment-based price discrimination matrix* where Government and Enterprise pay 10–60x what Midmarket pays. That's the entire game.
2. **COGS doesn't equal Units × Manufacturing Price.** Row 0: 1,618.5 units × $3 = $4,856, but COGS is $16,185 (exactly 50% of Gross Sales). COGS is being computed as a *percentage of revenue*, not as actual unit cost. That's a massive accounting anomaly worth interrogating — it means "Profit" in this dataset is a contribution-margin proxy, not real profit.
3. **Discounts column is all zero in the visible rows**, but Discount Band exists as a dimension. So somewhere in the full dataset there's a discount policy that's been turned off or selectively applied — which means there's a hidden lever.
4. **Fractional units sold** (1,618.5, 2,665.5) — these aren't widget sales. This is almost certainly a *services/license-equivalent* model where "units" are normalized contract values.

Now, three problem statements worthy of an MBA strategic finance role. Each is hard because it forces you to *interrogate the data's assumptions*, not just slice it.

---

## Problem 1 — The Price Discrimination Audit: Is Microsoft leaving money on the table, or destroying it?

**The setup.** You're the new Senior Manager of Strategic Finance. The CFO walks in: "Sales is pushing us to standardize pricing across segments to simplify GTM. Marketing wants to *expand* segment-based pricing further. Tell me which one is right by Friday."

**Why it's hard.** You'll discover the same SKU sells at radically different prices by segment ($15 to Midmarket, $350 to Government for Carretera). The instinct is to call this "price gouging Government" — but a real strategic finance answer requires you to:
- Compute *price elasticity by segment* (do high-price segments buy fewer units? The data lets you test this).
- Compute *contribution margin per segment per product* and compare to volume share — is the high-margin Government segment also the smallest-volume one (classic 80/20 trap)?
- Quantify the *revenue at risk* if you flatten pricing (the high-paying segments would defect or renegotiate; the low-paying ones won't suddenly pay more).
- Argue whether the price gap reflects genuine *willingness-to-pay differences* (procurement cycles, compliance overhead, switching costs) or whether it's a historical accident that competitors will arbitrage.

**The strategic answer.** Most MBAs will say "keep discrimination, it's optimal." The correct answer is more uncomfortable: the $350 Government price is almost certainly *unsustainable* once Government procurement officers benchmark it against the $15 Midmarket price (FOIA requests, GSA schedules, competitor intel decks all expose this). The right play is a **graduated convergence strategy** — hold Enterprise/Government pricing flat in nominal terms, let Midmarket pricing drift up 8–12% annually, and close the gap over 3–4 years while bundling justification (SLAs, compliance attestations) into the high-tier price. You preserve the rent extraction without the political/competitive risk of a flash audit.

---

## Problem 2 — The COGS Anomaly: Is the Profit number lying to the executive team?

**The setup.** You've been asked to build the FY budget. You pull this dataset and immediately notice: COGS is not Manufacturing Price × Units Sold. Row 0 should have COGS of ~$4,856 (1,618.5 × $3); the dataset shows $16,185 — exactly 50% of Gross Sales. Across rows, COGS appears to track revenue at 40–95% rather than tracking unit production cost. **Every "Profit" number every executive has ever seen from this report is wrong.**

**Why it's hard.** This is the kind of finding that ends careers — yours if you're wrong, someone else's if you're right. You have to:
- Prove the anomaly statistically (regress COGS on Units × Mfg Price vs. on Gross Sales — which fits better?).
- Hypothesize *why* — is COGS being booked as a standard-cost allocation? A transfer-pricing convention? A blended fully-loaded cost including allocated SG&A?
- Quantify the *delta* between reported profit and "true" contribution margin (Gross Sales − Units × Mfg Price). For row 5: reported profit $136,170 vs. true contribution $529,550 − $4,539 = $525,011. The dataset is *understating* unit economics by an order of magnitude.
- Decide what to tell the CFO: the company's product economics are dramatically better than reported, which means past decisions (discontinuing SKUs, exiting segments, capping sales comp) may have been made on bad data.

**The strategic answer.** This is a data-integrity finding masquerading as a finance question. The right move is *not* to publish a memo saying "the COGS is wrong." It's to: (1) replicate the anomaly in a controlled subset, (2) walk it to the Controller privately to understand whether COGS includes allocated overhead by design (in which case the issue is *labeling*, not math), (3) propose a dual-reporting framework — "Reported COGS" (current GAAP-aligned) and "Variable COGS" (Mfg Price × Units) — so commercial decisions use the latter and financial reporting uses the former. This is the move that protects you, helps the company, and demonstrates judgment over heroics.

---

## Problem 3 — The Portfolio Concentration Trap: Where would you cut, and why does that question have the wrong answer?

**The setup.** Board meeting in two weeks. Topic: "Rationalize the SKU and geography portfolio." The CEO has signaled she wants to exit underperforming segment-country-product cells. You have 4 countries × 5 segments × 6 products = up to 120 cells. The data will show that a small number of cells (probably ~15–20) deliver 80%+ of profit. The easy recommendation: cut the bottom 80 cells.

**Why it's hard and why the easy answer is wrong.**
- Many "low-profit" cells exist because they're *option value* — Mexico Small Business may lose money today but is the only beachhead into a market growing 3x faster than Germany.
- "Profit" in this dataset is corrupted (see Problem 2), so any cut decision based on the Profit column is built on sand.
- Segment-product fit matters: if you cut Government-Carretera in France, do you also lose Government-Montana in France because the same procurement contract bundles them? The dataset doesn't show contract linkages — but a real strategic finance lead has to flag what the data *cannot* answer.
- Fixed cost absorption: cutting 60% of cells doesn't cut 60% of overhead. The remaining cells become *less* profitable, not more, until you also restructure SG&A.

**The strategic answer.** Reframe the question the CEO asked. Instead of "which cells to cut," produce a **2x2 of strategic value (growth optionality, customer lock-in, competitive signaling) vs. current contribution margin**, and recommend three specific actions: (1) *Harvest* — 5–8 cells with high margin but low strategic value, take price, accept volume decline. (2) *Invest* — 3–5 cells with low current margin but high optionality, increase sales coverage. (3) *Exit* — only the cells that fail *both* axes, which will be far fewer than 80. Pair this with an explicit acknowledgment of what the data *can't* tell you (customer overlap, contract bundling, competitive response) and a $200K ask for a 6-week customer-level study to validate before any cuts. The strategic move is converting a "cut costs" mandate into a "reshape the portfolio with discipline" mandate — and buying the analytical runway to do it right.

---

**A meta-note on what makes these "MBA-grade":** Each problem requires you to (a) find a non-obvious pattern in the data, (b) recognize what the data *can't* answer, (c) navigate organizational politics in your recommendation, and (d) deliver an answer that's defensible *and* slightly contrarian. If your final memo on any of these reads like a Power BI dashboard summary, you've failed the assignment. The dashboards describe; strategic finance *decides*.

Want me to build out a working Python/Excel model for any one of these — the elasticity regression for #1, the COGS reconciliation for #2, or the 2x2 portfolio framework for #3?