# ROADMAP — Combating Popularity Bias in LightGCN

Source of truth for scope: `longtail_lightgcn_proposal.pdf`. Working rules: `CLAUDE.md`.

Every milestone is reviewable on its own. Do not start milestone N+1 until N is reviewed and approved.
Each milestone lists **Decision points** — these are NOT decided here. Before implementing, Claude presents 2-3 options with tradeoffs and waits for a decision. Each milestone also needs a short plan approved before any code.

Decisions made so far are recorded in the **Decision log** at the bottom.

---

## M1 — Data pipeline

**Goal:** Load MovieLens-1M, binarize to implicit feedback, split into train/val/test, and build the normalized sparse adjacency `D^-1/2 A D^-1/2` in PyTorch.

**In scope:** loading, binarization, ID remapping, splitting, adjacency construction, item metadata loading (genre, year) as raw tables only.
**Out of scope:** the model, negative sampling, metrics, any intervention (including degree-discounting the adjacency).

**Decision points**
- Binarization rule (all ratings as positives vs. threshold such as rating >= 4).
- Split strategy (random per-interaction, leave-one-out, per-user ratio, temporal), including how "various splits" in the proposal is interpreted.
- Item filtering / k-core. This changes the item count and interacts with Intervention C, because cold items must exist in the data for C to matter.
- Data storage format and how raw data is obtained (download script vs. committed files).
- Adjacency representation: full (U+I)x(U+I) bipartite sparse matrix vs. separate user-item blocks. Sparse format (COO/CSR).

**Tests / done when**
- Counts of users, items, interactions printed and sanity-checked.
- No train/val/test leakage (tested).
- Adjacency is symmetric, and normalization checked on a tiny hand-computed example.

---

## M2 — Baseline LightGCN

**Goal:** From-scratch LightGCN: embedding tables, K propagation layers, mean layer combination, BPR loss with negative sampling, plain training loop. No interventions.

**In scope:** model, BPR loss, negative sampler, training loop, a minimal Recall@20 evaluation sufficient for the sanity check.
**Out of scope:** Gini / coverage / popularity metrics (M3), all interventions, config flags for A/B/C.

**Decision points**
- Negative sampling scheme (uniform per positive, number of negatives, sampling on the fly vs. precomputed).
- Hyperparameters: embedding dim, K, learning rate, L2 regularization, batch size, epochs, early stopping criterion.
- Embedding initialization (the LightGCN paper uses Xavier-style / normal init).
- Evaluation protocol for ranking (full ranking over all items, masking train items).

**Tests / done when**
- Manual propagation on a tiny 3-user, 3-item graph matches the model's output (hand-computed expected values).
- Recall@20 sanity check. **Flag:** the LightGCN paper did not use ML-1M, so there is no directly comparable published number. We will need to decide what "sanity" means (e.g., ranges reported in other papers on ML-1M, or comparison against a popularity baseline and a simple MF-BPR baseline we write ourselves).
- Training loss decreases; baseline clearly beats a most-popular recommender or the discrepancy is explained.

---

## M3 — Metrics

**Goal:** Recall@K, NDCG@K, Coverage, Average Recommendation Popularity, Gini — written and unit-tested on toy data.

**In scope:** metric functions and their tests only.
**Out of scope:** the grid, plots, head/mid/tail breakdown (M6).

**Decision points**
- K value(s) (proposal does not specify; M2 used 20).
- Gini definition: computed over item recommendation counts across all items in the catalog (including items recommended 0 times) vs. only recommended items; formula/sorting variant.
- Average Recommendation Popularity definition (mean train-degree of recommended items, per user then averaged vs. pooled).
- Coverage: share of the whole catalog vs. share of items in the eligible set.

**Tests / done when**
- Toy cases with known answers: Gini = 0 for perfectly uniform exposure, near 1 when one item gets everything; coverage, recall and NDCG hand-computed.
- Edge cases: items never recommended, users with no test items.

---

## M4 — Interventions (one at a time, each its own sub-review)

Order: A, then B, then C. Each sub-milestone ends with the reproduction test before the next begins.

### M4a — Intervention A: Degree-Discounted Aggregation
Per-item factor `|N(i)|^-γ` on the item-update direction of propagation (Eq. 1).
**Decision points**
- Implementation: a modified, asymmetric adjacency (user->item block scaled) built once vs. scaling inside the forward pass. Step 4 of the notes suggests the adjacency.
- Whether the discount applies at every layer (as Eq. 1 reads) and on the item-update direction only (as written) — note this makes the propagation matrix non-symmetric, which should be understood and stated.
- Whether to re-normalize after discounting.

### M4b — Intervention B: Popularity-Aware Loss Re-weighting
Per-sample weight `w(i) = pop(i)^-β` on the positive item (Eq. 2).
**Decision points**
- `pop(i)` definition: training-set degree (raw count), computed on train only.
- Weight normalization (so the effective loss scale / learning rate does not shift with β), and loss reduction (sum vs. mean). The proposal writes a plain sum with no normalization.

### M4c — Intervention C: Content-Based Cold-Item Initialization
`e_i^(0) = e_i^ID + f_θ(x_i)`, `x_i` = genre one-hot + normalized release year, `f_θ` a lightweight MLP (Eq. 3).
**Decision points**
- MLP architecture (depth, width, activation, dropout).
- Release-year source (parsed from title) and normalization; handling missing years.
- Whether `f_θ` shares a learning rate / regularization with the ID embeddings.
- How to evaluate cold-item benefit given the split strategy chosen in M1.

**Cross-cutting (done at the end of each sub-milestone)**
- Config flag per intervention; γ=0, β=0, and C-off must reproduce baseline **exactly** (test bitwise or within a stated tolerance, with the same seed).
- Decision: how configs are represented (dataclass, YAML, CLI args).

---

## M5 — Experiment grid

**Goal:** Run the full ablation.

- 2^3 = 8 on/off combinations of A, B, C.
- Sweep γ and β (e.g., {0, 0.25, 0.5, 0.75, 1}), 3-5 seeds each.

**Decision points**
- Grid shape: sweeping γ and β independently vs. jointly (the full joint cross is 5x5=25 with C on/off = 50 configs, x seeds). Compute budget should be estimated from a single timed run first.
- Number of seeds (3 vs. 5).
- Hyperparameter selection: tune on validation only; test touched once for final numbers.
- Compute platform (Colab T4 / Kaggle / local) and result logging format.
- Checkpointing/resumability for long sweeps.

**Done when:** every config × seed has a logged result row (CSV/JSONL) with all M3 metrics; run is reproducible from a config + seed.

---

## M6 — Analysis & report

**Goal:** Produce the results for the final report and README.

- Recall-vs-Gini scatter across all configurations, with the Pareto frontier.
- Head / mid / tail item breakdown of recall and exposure.
- Mean ± std over seeds.
- Test of the proposal's hypothesis: does the combined method give a better accuracy-bias tradeoff than base LightGCN or any single intervention? Report honestly if it does not.

**Decision points**
- Head/mid/tail cutoffs (e.g., top 20% / middle / bottom by train popularity, or a Pareto-style 80% interaction mass).
- Plotting library and figure style.
- Whether to report significance tests across seeds.

---

## Open questions / flags to resolve with the user

1. **Item count mismatch:** proposal says ~3,900 movies; the notes say ~3,700. ML-1M has 3,883 movies in `movies.dat` but only ~3,706 appear in `ratings.dat`. Confirm which the project should treat as the catalog (affects Coverage and Gini denominators).
2. **No published ML-1M LightGCN number** exists in the original paper, so the M2 sanity check needs a defined reference.
3. **Eq. 1 asymmetry:** discounting only the item-update direction makes the propagation matrix non-symmetric. Confirm this is intended in the proposal before M4a.
4. **Cold-item claim:** the proposal motivates C with cold items, but a standard random split leaves few true cold items in test. M1's split choice must account for this.
5. **Grid size vs. "a few hours":** depends on the sweep design in M5.

---

## Decision log

_Empty. Each decision is added here, with date and rationale, once the user has made it._

| Date | Milestone | Decision | Options considered | Rationale |
|------|-----------|----------|--------------------|-----------|
