# VIVA.md: questions I expect, with answers

These were written during the build, as each decision was made. Every number is quoted from `results/metrics.json`, which `python -m src.evaluate --report` regenerates. If a re-run changes a number, `/metrics` on the live URL shows the current value. Questions marked **[after Day-2 ingest]** get their numbers once the real labels are in (`metrics.json → real_validation`).

---

## A. Data and problem framing

**Q1. Your data is synthetic. Why should I believe any of your numbers?**
The synthetic numbers show the *pipeline* is correct: a time-based split, honest baselines, calibrated confidence, and a detector scored against known ground truth. They don't claim real-world accuracy. That claim comes from the **real validation set**: 300 real bank-statement strings labelled by three people, plus 30 anchor strings that all three labelled, so I can measure how much humans agree. The gap between synthetic and real is reported as a finding. I also wrote my expected real-data accuracy in DECISIONS.md *before* seeing the labels, and committed it first, so git history shows I didn't move the goalposts.

**Q2. How did you stop the generator from making the task trivially easy?**
Three kinds of ambiguity. (1) Shared platforms: Amazon Pay, Paytm, Swiggy and others appear under several categories. (2) Payments to individuals, drawn from one shared name pool across categories, so an auto driver and a kirana owner look identical. (3) Surface noise: legal entity names, truncation, typos, casing. On top of that there's a gate: if synthetic accuracy is above 92%, the script prints `LEAKAGE SUSPECTED` and exits with an error, and I make the *generator* harder. The gate actually fired in the *other* direction. The first generator scored 0.729, below the 0.78 floor. One logged round, halving the person-payment share, brought it to 0.793. I stopped there and never touched the model's hyperparameters.

**Q3. Why mask digits?**
Privacy and robustness. Real strings contain phone numbers (`UPI/98xxxxxxxx/...`) and account references. Every run of 4 or more digits becomes `#` before a string is written to any file. The model applies the same mask to all input, so it can't learn from reference numbers, which are random and carry no category signal.

## B. Classifier

**Q4. Why character n-grams and not word n-grams?**
Merchant strings aren't language. `SWIGGY`, `SWIGY`, `swiggy@icici` and `BUNDL TECHNOLOGIES` are word-level strangers. Truncation (`ZOMATO ONLINE BANGA`) and concatenation (`bundltechnol@upi`) keep inventing tokens the model has never seen. Character 2–4-grams (`SWI`, `WIG`, `IGG`) survive all of that, because most of a string's n-grams are shared with its variants. I used `char_wb`, so the n-grams also mark word boundaries.

**Q5. Why TF-IDF and not embeddings (e.g. sentence-transformers)?**
Embeddings are trained on natural language, and these strings aren't natural language. They'd also add a large model to a CPU-only container that has to cold-start on a phone demo. TF-IDF + logistic regression trains in seconds, predicts in milliseconds, is deterministic (so the build can verify it byte-for-byte), and every coefficient can be read. With about 7.6k training strings, I'd need evidence that embeddings help before paying that cost. That would be the next experiment, measured on the real validation set.

**Q6. Why a time-based split? Why never random?**
In deployment the model always predicts *future* transactions from past ones. A random split puts, say, March 2026 strings into training while testing on February 2026. That's leakage from the future, and it hides drift such as new merchants or new formats. The split is at 2026-04-05: 7,597 training rows before it, 1,911 test rows after it. The forecaster holds out the last 8 weeks for the same reason.

**Q7. Why `class_weight="balanced"`?**
The classes are imbalanced: Food & Dining has far more rows than Education. Unweighted log-loss is dominated by big classes, so the model learns to ignore small ones and still looks accurate. Balanced weights scale each class's loss by the inverse of its frequency. That's also why I report **macro-F1**, which averages classes equally, alongside accuracy, which is dominated by big classes.

**Q8. What does a row of the confusion matrix mean?**
A row is every test transaction whose *true* category is that row. The columns show where the model sent them. The diagonal cell divided by the row total is that class's **recall**. Example: the Groceries row is 28 → Food & Dining, 162 correct, 29 → Shopping, 23 → Transfers, and so on, which gives recall 0.596. Those off-diagonal cells are exactly the designed ambiguity: Swiggy/Zomato quick-commerce orders read as food, Flipkart/Amazon grocery orders read as shopping, and kirana payments to individuals read as transfers. A *column* divided by its total is **precision**.

**Q9. Is RandomForest actually better than LogisticRegression?**
Not on the metric that decides it. RF has the higher macro-F1 (0.760 [0.735, 0.784] vs 0.746 [0.721, 0.768]), and the confidence intervals overlap heavily. But the **exact McNemar test** on the same 1,911 test rows found 71 rows where only LR was right, against 42 where only RF was right, **p = 0.0081**. LR makes significantly fewer errors overall (accuracy 0.793 vs 0.778). RF buys a little small-class recall at the cost of more total errors. The rule was fixed in code before the comparison: p ≥ 0.05 ships LR; p < 0.05 ships the model the test favours. So LR ships.

**Q10. Why McNemar and not just "higher number wins"?**
Two models scored on the *same* rows are paired, so their errors are correlated. Comparing two accuracies as if they were independent ignores that. McNemar looks only at the **discordant** pairs (one model right, the other wrong). Under "equal error rates", each discordant pair is a coin flip, so the count is Binomial(n, 0.5). I implemented the exact binomial version with `scipy.stats.binomtest`, which is about ten lines in `src/stats.py`.

**Q11. What are the bootstrap confidence intervals?**
The test set is resampled with replacement 1,000 times (seeded), and accuracy and macro-F1 are recomputed each time. The 2.5th and 97.5th percentiles form the 95% CI. LR accuracy is 0.793 [0.774, 0.811]. The lower end dips below my 0.78 band. I report that as it is rather than tuning the generator further to push it up.

**Q12. What does a confidence of 0.8 from your model actually mean?**
Less than it should: the model is **underconfident**. The reliability curve shows that in the 0.7–0.8 bin (mean confidence 0.751), actual accuracy is 0.937, and in the 0.8–0.9 bin (mean 0.847), it's 0.994. ECE is 0.160. Balanced class weights and L2 regularisation spread probability mass across 10 classes. That's why the abstain threshold is chosen from **observed** accuracy, not the raw probability: at 0.65, the model answers on 56.8% of test transactions with 95.4% accuracy on those, and flags the rest `uncertain: true`. Temperature scaling or isotonic calibration on a separate time slice would be the fix.

**Q13. What's in `results/error_analysis.md`, and what did it teach you?**
It lists the 15 highest-confidence wrong predictions. Nearly all are shared-platform strings: Swiggy/Zomato strings whose true label is Groceries (Instamart/Blinkit orders) are predicted Food, and Flipkart grocery orders are predicted Shopping. The model can't know which sub-service was used, because the bank string doesn't say. That's an information limit, not a modelling bug. The fix is a different feature (amount, time of day, basket history), not a bigger model.

## C. Anomaly detection

**Q14. How does the detector work?**
For each debit, it computes a z-score against the **same category's** transactions in the **previous 28 days**. The current day is excluded, so a spike can't inflate its own baseline and partly hide itself. It needs at least 5 past transactions, and the first 28 days are warm-up. There's nothing to train, which is appropriate because real users have no anomaly labels.

**Q15. Precision vs recall for anomalies: which matters more here?**
For a personal-finance nudge, **precision**. Every false alarm teaches the user to ignore the alerts. For fraud detection you'd favour recall. My sweep shows the trade-off: raw-z at |z| > 2 catches 72.8% of anomalies at 27.9% precision, and at |z| > 4 it catches 47.3% at 45.8% precision. The brief's |z| > 3 gives P = 0.368, R = 0.592. I report the whole sweep, not just the best point. The base rate is 2.03%, so random flagging would have precision around 0.02, and every row is well above that.

**Q16. You expected log-z to win. What happened?**
It lost: log-z AUC-PR 0.335 vs raw-z 0.386. My reasoning was that amounts are lognormal, so the log should make the z-score behave. What I missed: the injected anomalies multiply the amount by 4–10×. In log space that's a shift of only log(4)–log(10) ≈ 1.4–2.3, which is just 1.5–3.8 σ in categories with a wide spread. On the raw scale the jump is huge. The served detector follows the rule I wrote *before* seeing results (highest AUC-PR): raw-z.

**Q17. Where does the detector fail?**
Per-category precision shows it. **Transfers 0.125** is extremely heavy-tailed. **Bills & Utilities 0.185** has a design flaw I found while testing the API: the 28-day window is *shorter than a monthly billing cycle*. Last month's credit-card or rent payment has always left the window by the time this month's arrives, so every monthly bill is compared against a month of small recharges and looks like a spike. The fix is a window of 35 days or more, or a per-merchant baseline. I didn't change it: the brief fixes the window at 28 days, and changing it after seeing results would be tuning to the evaluation. It's documented as a limitation.

## D. Forecasting

**Q18. Why does the naive baseline matter?**
A forecast is only useful if it beats "next week = last week". Otherwise the model adds complexity for nothing. I report four models, and the baselines are allowed to win. **They do.** The *category mean*, a constant, is the per-category winner in 5 of 10 categories. Linear regression wins 3, and naive and seasonal-naive win one each. Linear has the lowest *average* MAE (₹4,251 vs ₹4,353 for the mean), but the constant beats it in 7 of the 10 categories individually. Most categories' weekly spend is close to noise around a mean, and lag features just chase last week's noise. Each category serves its own winner, and `/forecast` shows which model was used and its holdout MAE.

**Q19. Why MAPE and not just MAE?**
MAE is in rupees, so it's interpretable but not comparable across categories: a ₹2,000 error on Transfers (weekly spend in the tens of thousands) is excellent, while on Health it's terrible. MAPE is scale-free. But MAPE is undefined when actual = 0 and explodes near zero, so I compute it only on weeks with non-zero spend and store how many weeks were used. The mean-baseline's MAPE (103%) is *worse* than linear's (74%) even though its MAE is close. MAPE punishes a constant heavily in low-spend weeks, which is exactly why you report both.

**Q20. How are the forecast intervals built, and why aren't they 80% by construction?**
They use the empirical 10th/90th percentile of one-step residuals from a **separate 16-week calibration window** just before the holdout, with the models fit only on data before that window. Coverage is then measured on the 8 holdout weeks. If I had taken the quantiles from the holdout residuals themselves, coverage would be ~80% automatically and prove nothing. The result is mean coverage 0.800. That's a fair result, not a forced one: per category it ranges from 0.75 to 0.88. Caveat: with 8 weeks, coverage moves in steps of 0.125, so one week either way changes it a lot.

**Q21. Your forecast interval might cover 62%, not 80%. Why would that happen?**
Ours landed at 0.800 on the mean, but it could easily have come out low, and the reasons are worth knowing. (1) **Non-stationarity:** if the holdout period (July–August) is more volatile than the calibration window (March–June), for example a festival or a big purchase, residuals get bigger and a fixed-width interval under-covers. (2) **Small calibration sample:** 16 residuals give noisy 10th/90th percentiles. (3) **Heavy tails:** spend is lognormal, so a few extreme weeks dominate. I'd report 62% as the result, just as I report 0.800. The fix is a longer calibration window or conformal methods with rolling recalibration, not widening the interval until the number looks right.

**Q22. What's "k/8", and what does "not robust" mean?**
It counts how many of the 8 holdout weeks the winner beat the reference baseline (seasonal-naive, or naive when the winner *is* seasonal-naive). A model can win on average MAE through one lucky week. Miscellaneous is flagged: its winner beat the reference in only 4 of 8 weeks. It's a sanity check in the spirit of the Diebold-Mariano test, but it isn't a formal test at n = 8.

## E. The LLM

**Q23. What does the LLM do, and what does it NOT do?**
It **only narrates**. It receives the JSON output of the three models and writes 2–4 plain-English sentences. It never sees raw transactions, and it never calculates, predicts or estimates. Enforcement is three layers: (1) the system prompt forbids computing or inventing numbers and requires hedging on `uncertain: true` items; (2) a **number-guard** extracts every number from the reply and rejects the reply if any number isn't in the input (allowing rounding and fraction→percent); (3) on rejection, timeout (5 s) or any error, a deterministic template built from the same JSON is returned instead. `/advice` can't 500. The smoke test simulates a hung network (an 8 s sleep) and checks that the fallback arrives within the timeout.

**Q24. Why cache LLM responses, and what's the cache key?**
The key is sha256 of (model name | prompt version | canonical JSON payload). The same numbers get the same explanation, instantly, at zero cost, which matters on an untrusted demo network. Changing the model or the prompt invalidates the cache automatically. `/health` exposes cache hits and misses, fallback count, token counts and an estimated INR cost.

## F. Real data **[after Day-2 ingest]**

**Q25. Why is your real-data accuracy lower than your test accuracy?**
*Fill in from `real_validation`: synthetic 0.793 vs real ___, gap ___.* The structural reasons: (1) **Vocabulary shift.** Real statements contain merchants, banks and formats my generator never produced, and char n-grams generalise to spelling variants, not to unseen brands. (2) **More person-to-person UPI.** Real Indian statements have a large share of payments to individuals and small shops with personal VPAs, which are ambiguous from text by construction. (3) **Label noise.** Human labellers disagree (see Q26). Synthetic labels are exact. Compare the result against the pre-registered range in DECISIONS.md: say whether it fell inside, and which categories degraded most versus the prediction.

**Q26. What's your human annotation ceiling, and how does the model compare?**
*Fill in: Fleiss' κ = ___, mean pairwise agreement = ___.* Thirty anchor strings were labelled by all three people. κ corrects for chance agreement, so it isn't on the accuracy scale. The fair ceiling for accuracy is **mean pairwise agreement**: how often two humans assign the same label. If the model is within a few points of it, the remaining errors are ambiguity in the strings themselves, not model weakness. The pipeline was verified end to end with simulated labellers.

**Q27. What would you do differently with real transaction data?**
(1) Train on real labelled strings, even a few thousand, and keep the synthetic data only for augmentation. (2) Add non-text features: amount, day of month, recurrence ("same payee, same amount, monthly" → a bill), and payee history. That directly fixes the rent-vs-transfer and Instamart-vs-Swiggy confusions. (3) An anomaly baseline per merchant, with a window longer than one billing cycle. (4) Calibrate probabilities (isotonic, on a held-out time slice). (5) Active learning: send `uncertain: true` strings to the user for one-tap correction, which is the cheapest source of real labels.

## G. Engineering

**Q28. What would break at 10× data?**
Probably not training: TF-IDF + LR is linear in the number of rows, so about 95k strings should still train in minutes. I haven't measured it. The things that break: (1) `/summary` and `/anomalies` load the whole CSV into memory at startup, so at 10× users (not rows) that needs a database and per-user partitioning. (2) Anomaly scores are computed once at build time over a static file, so real-time data needs incremental rolling statistics. (3) The RandomForest comparison and 1,000-resample bootstrap would make `evaluate --report` slow; the rule already drops to 500 resamples if runtime passes 3 minutes. (4) The TF-IDF vocabulary grows with new merchants, so it needs `max_features` or a hashing vectorizer. (5) The JSON-file LLM cache needs an external store (Redis/Firestore) once there's more than one Cloud Run instance.

**Q29. How do you know the deployed model is the one you evaluated?**
The Docker build retrains from source (seeded, with pinned library versions), then `src/verify_build.py` recomputes test accuracy and macro-F1 and compares them with the committed `results/metrics.json`. A mismatch above 0.005 **fails the build**. `/health` shows `metrics_verified: true` and the model's sha256. In a clean-checkout simulation the rebuilt model's hash was identical (`1f32b709ffca`), and the deltas were 0.0000.

**Q30. Why is `results/metrics.json` the single source of truth?**
Because hand-copied numbers drift. `evaluate --report` writes metrics.json, and then renders every README table *from* it, between markers. Even the "what didn't work" bullets are generated from conditions in the data. Two consecutive runs produce a byte-identical metrics.json, and I checked that. `/metrics` serves the same file live, so an examiner can compare the README against production.
