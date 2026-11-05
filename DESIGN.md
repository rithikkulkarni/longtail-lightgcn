# DESIGN.md — Design Decision Log

Every meaningful design decision for this project, with the options considered, the tradeoffs, and why we chose what we chose. Written so each entry can be reused to explain and defend the decision in an interview.

**Conventions**
- Entries are numbered `D-xx` and are never deleted. If a decision changes, add a new entry and mark the old one `Superseded by D-yy`.
- Status: **Decided**, **Open** (waiting on the user), **Parked** (deferred to a later milestone), **Revisit** (decided, but to be re-examined with data).
- The decision-maker is always the user. Claude presents options and implements.
- Scope reference: `longtail_lightgcn_proposal.pdf`; milestones in `ROADMAP.md`.

---

## M1 — Data pipeline

### D-01 · Binarization — **Decided** (2026-10-16)
- **Choice:** every rating becomes an edge (rating >= 1).
- **Options:** (A) all ratings -> 1, ~1.0M edges; (B) rating >= 4, ~575K edges; (C) rating >= 3, ~830K edges.
- **Why:** the study concerns whether a user watched an item, not whether they liked it. It matches the proposal's "roughly 1 million ratings" and the usual implicit-feedback convention in the LightGCN literature.
- **Tradeoff accepted:** a 1-star rating becomes a "positive", so the model learns what users chose to watch, not what they liked.
- **Impact:** edges define degrees, so this sets the popularity distribution used by Interventions A and B and by Gini. Results are not directly comparable to a thresholded variant.

### D-02 · Split strategy — **Decided** (2026-10-16)
- **Choice:** per-user random split, 80/10/10 train/val/test. N repeated splits, one per seed, default N = 5 (may change to 3 or more later).
- **Options:** (A) per-user random; (B) leave-one-out by timestamp; (C) global temporal split. For multiple splits: repeated random, k-fold per user, or a single split.
- **Why:** standard LightGCN protocol, so the baseline is comparable and easy to defend. Repeated splits guard against conclusions that depend on one lucky split.
- **Tradeoffs accepted:** ignores time (the model can train on "future" interactions); repeated random test sets can overlap across seeds, so splits are not independent like k-fold folds; items with few ratings can end up with zero train edges (see D-07). Multiplies M5 compute (splits x configs x model seeds).

### D-03 · No filtering — **Decided** (2026-10-16)
- **Choice:** no k-core or other item/user filtering.
- **Why:** low-degree items are exactly what this project studies, and ML-1M already guarantees >= 20 ratings per user.
- **Tradeoff accepted:** noisy embeddings for items with a handful of edges, and zero-train-degree items after the split.

### D-04 · Item catalog — **Decided** (2026-10-16)
- **Choice:** catalog = the 3,706 movies with at least one rating. The 177 unrated movies are removed from the graph and catalog but kept in `items.csv` with an `in_catalog` flag. An ID mapping (original movie ID -> index) is saved.
- **Options:** keep all 3,883 as isolated nodes vs. remove the unrated ones.
- **Why:** unrated items can never be evaluated (no held-out edges), they distort Coverage and Gini denominators, and they have degree 0. They also can't be used to measure whether Intervention C helps, since there is no ground truth for them.
- **Reversibility:** easy. The metadata and mapping are preserved.
- **Note:** the proposal says ~3,900 movies, which counts the full `movies.dat`; the model uses 3,706.

### D-05 · Data storage — **Decided** (2026-10-16)
- **Choice:** a download script fetches ML-1M into `data/raw/`; preprocessing writes `data/processed/split_{seed}.npz` (edge arrays) and `items.csv` (metadata). `data/` is gitignored.
- **Options:** download + preprocess scripts; parse and split in memory on every run; commit raw data.
- **Why:** reproducible on Kaggle with one script, every experiment uses the identical saved split, and loads are fast across the M5 grid.
- **Formats:** `.npz` for edges (integer arrays, fast, no extra dependency); CSV for item metadata (tiny, human-readable). Alternative was Parquet/CSV for everything (more inspectable, slower, needs `pyarrow`).

### D-06 · Adjacency representation — **Decided** (2026-10-16)
- **Choice:** full (U+I) x (U+I) symmetric normalized adjacency `D^-1/2 A D^-1/2`, stored as a PyTorch **COO** sparse tensor, built from **train edges only**.
- **Options:** full matrix vs. two separate blocks R and R^T; COO vs. CSR.
- **Why:** matches the LightGCN paper and is standard. Intervention A (Eq. 1) is a row scaling `S·Â` where `S` is diagonal (1 for users, `|N(i)|^-γ` for items), so it fits cleanly, and γ=0 gives identity.
- **Correction on record:** Claude initially suggested the full matrix made A awkward; that was wrong.
- **COO vs. CSR:** COO is simplest to build and test; the speed difference is negligible at ~2M nonzeros. Claude was unsure how fully PyTorch supports CSR in backprop, so verify before ever switching.
- **Why train-only:** val/test edges in the adjacency would leak into propagation and inflate Recall.

### D-07 · Zero-train-degree nodes — **Decided, Revisit** (2026-10-16)
- **Choice:** guarded normalization: degree 0 gets a normalization factor of 0 (no NaN or inf). The data build reports how many items have zero train edges per split.
- **Options:** (A) guard in the math, keep the split purely random; (B) constrain the split so every item has at least one train edge.
- **Why:** keeps the split purely random and avoids hiding cold-in-train items, which Intervention C is meant to help.
- **Observed (M1, 5 splits):** 114 items have exactly 1 rating overall; 22-31 of 3,706 items (<1%) have zero train edges per split.
- **Revisit when:** we see the real counts. Intervention A (`|N(i)|^-γ`) and B (`pop(i)^-β`) will also need a degree-0 rule; decide in M4.

### D-08 · Parsing library — **Decided** (2026-10-16)
- **Choice:** `pandas`.
- **Options:** `pandas` (less code, handles the `::` separator and encodings, one more dependency) vs. plain Python/`numpy` (nothing extra, more manual). Claude leans pandas.

### D-09 · Per-user rounding rule — **Decided** (2026-10-16)
- **Choice:** half-up rounding: `n_val = n_test = int(0.1·n + 0.5)`; train gets the rest.
- **Why:** keeps the split close to 10% with no systematic shortfall, and avoids Python's round-half-to-even quirk (`round(2.5) == 2`, `round(3.5) == 4`).
- **Note:** changing the rule later changes every saved split.
- **Options:** `n_val = n_test = round(0.1·n)` vs. `floor(0.1·n)`, train gets the rest. `round` keeps the ratio closer to 10%; `floor` is simpler to explain. Effect on results is expected to be tiny.

### D-10 · Project layout — **Decided** (2026-10-16)
- **Choice:** `src/longtail_lightgcn/` with a minimal `pyproject.toml` and `pip install -e .`.
- **Options:** `src/longtail_lightgcn/` with a minimal `pyproject.toml` and `pip install -e .` (clean imports from tests and Kaggle notebooks) vs. a flat layout with no packaging (simpler, messier once notebooks import from it). Claude leans `src/`.

- **Note (2026-10-16):** PyTorch emits "Sparse invariant checks are implicitly disabled" when `build_norm_adj` creates the sparse tensor. Harmless advisory; user chose to leave it. Options if it ever matters: pass `check_invariants=True` (cheap, adds a safety net) or filter the warning.

---

## M2 — Baseline LightGCN

### D-12 · Hyperparameters — **Decided** (2026-10-16)
- **Choice:** the authors' README settings: embedding dim 64, K = 3 layers, Adam, lr 0.001, L2 weight 1e-4, batch size 2048, up to 1000 epochs, mean layer combination (layer 0 included, weights 1/(K+1)).
- **Options:** (1) paper/README defaults; (2) small validation sweep for the baseline.
- **Verified against** `kuandeng/LightGCN` README commands. Note the repo's *argparse defaults* differ (lr 0.01, batch 1024, decay 1e-5, 4 layers); the README commands are the intended settings and match the paper.
- **Why:** closest to "unmodified LightGCN" in the proposal; least tuning needed.
- **Tradeoff / Revisit:** these were tuned on Gowalla / Yelp2018 / Amazon-Book, not ML-1M, so may be suboptimal. Fairness question for M5: if the baseline is tuned, the interventions must get the same tuning budget, otherwise the comparison favors one side.
- **Early stopping (decided 2026-10-16):** authors' schedule: evaluate every 20 epochs, stop after 5 consecutive evaluations without improvement. The authors use the **test** set for this; we use **validation** Recall@20 and touch test only at the end.
- **Amendment for the first full baseline run (2026-10-16, user's choice "B"):** evaluate every 5 epochs with patience 20 (`--eval_every 5 --patience 20`). Same effective patience as the authors (100 epochs after the peak) but a finer validation curve. Purpose: learn the shape of the curve (when it peaks, how noisy) before choosing a cheaper stopping rule for the M5 grid.
- **Options considered:** (A) authors' 20 x 5; (B) 5 x 20; (C) 5 x 5, which cuts patience to 25 epochs and saves ~75 epochs (~26 min at 21 s/epoch) per run but risks stopping before the true peak and so weakening the control.
- **Rule for later:** whatever stopping rule is used for the final experiments must be identical for every configuration. Revisit in M5 by replaying the logged curve.
- **Measured (2026-10-16, Kaggle GPU):** ~21 s/epoch (vs. ~341 s on CPU). Minimum run ~120 epochs (~45 min); worst case 1,000 epochs (~6 h). This makes the full M5 grid (configs x splits x seeds) expensive; see P-06.

### D-13 · Evaluation protocol — **Decided** (2026-10-16)
- **Choice:** full ranking over all 3,706 items; mask train items for validation, train + val items for test; K = 20 for now (final K for all metrics is an M3 decision).
- **Options:** full ranking vs. sampled evaluation (positive vs. ~100 random negatives).
- **Why:** unbiased and cheap at this scale; sampled evaluation is known to give misleading comparisons.

### D-14 · Embedding initialization — **Decided** (2026-10-16)
- **Choice:** Normal(0, 0.01), as in the authors' official TensorFlow code.
- **Sources disagree:** paper text says Xavier; official code uses `random_normal_initializer(stddev=0.01)` (Xavier only in a comment); the popular third-party PyTorch port (`gusye1234/LightGCN-PyTorch`) uses Normal(0, 0.1). For our 9,746 x 64 table, Xavier-uniform has std ~0.014, so the paper's text and the official code agree in scale.
- **Why it matters:** tail items receive few updates and stay near their initial values for long, so init scale influences where they rank. Keep it fixed across all configurations.
- **Related:** Intervention C modifies layer-0 item embeddings (`e_ID + f_theta(x)`), so this choice defines the ID part of C.

### D-15 · L2 regularization form — **Decided** (2026-10-16)
- **Choice (D1):** paper form: penalize only layer-0 embeddings of the users, positives and negatives in the current batch: `lambda * 0.5 * (||e_u||^2 + ||e_i||^2 + ||e_j||^2) / batch_size`. Verified against the official code.
- **Option rejected (D2):** optimizer weight decay over all embedding rows every step.
- **Effect:** regularization strength scales with how often a row is touched, i.e. with its data signal ("per-example" L2). D2 would shrink rarely-seen tail items toward zero at the same per-step rate as popular ones, adding popularity bias to the baseline itself (interaction with Adam is subtle and was not tested).

### D-16 · Negative sampling — **Decided** (2026-10-16)
- **Choice (A1):** one negative per positive (one per train edge per epoch), drawn uniformly from all items and redrawn if the user has it in **train**. Never rejects against val/test (that would leak).
- **Guard:** the sampler raises a clear `ValueError` at construction if any user has interacted with every item (no valid negative exists; the redraw loop would never end). Will never fire on ML-1M (max ~1,850 train items of 3,706); tested with a synthetic user.
- **Options rejected:** A2 uniform with no check (false negatives for heavy users); A3 popularity-proportional (acts as an uncontrolled debiasing intervention inside the baseline; popular items are drawn as negatives more, which pushes them down).
- **Known property:** with uniform negatives a popular item is pushed up ~deg(i) times per epoch but down only ~216 times (800K draws / 3,706 items), a known source of popularity bias in BPR that exists in the baseline before any intervention.
- **Unavoidable noise:** items that would be liked (val/test items, never-seen items) can still be drawn as negatives (false negatives).

### D-17 · Sanity check for baseline Recall@20 — **Decided** (2026-10-16)
- **Choice (F2):** compare against ranges reported in other papers for ML-1M with random splits; a loose check only.
- **Caveat:** the original LightGCN paper did not use ML-1M, and published numbers vary with split, protocol and K, so this can catch gross bugs but not small ones. Not adding extra baseline models (popularity, MF-BPR) in M2.
- **Hand check still required:** the 3-user, 3-item propagation test.

### D-18 · Recall@K formula — **Decided** (2026-10-16)
- **Choice:** standard form: per user, `hits in top-K / number of held-out items`, averaged over users.
- **Alternative:** divide by `min(K, number of held-out items)`, which differs only for users with more than K held-out items. Final metric definitions are an M3 decision; M2 uses this minimal form.

### D-19 · Config handling — **Decided** (2026-10-16)
- **Choice:** a plain `dataclass` holding the paper's defaults (D-12), so hyperparameters can be changed in one place.
- **Deferred:** how configs are represented across the whole experiment grid (YAML / CLI / dataclass) is an M4/M5 question.

### D-20 · Kaggle notebook in M2 — **Decided** (2026-10-16)
- **Choice:** the thin Kaggle experiment notebook is part of M2, so the baseline run happens on Kaggle's GPU and the user learns the end-to-end process before the research experiments start. (ROADMAP M2 scope widened accordingly.)

---

## M3 — Metrics

Metrics are exactly the five in the proposal: Recall, NDCG, Coverage, Average Recommendation Popularity (ARP), Gini. Implemented in `metrics.py`.

### D-21 · K — **Decided** (2026-11-04)
- **Choice:** K = 20 only. **Options:** K=20; K=10 and 20; K=10, 20, 50.
- **Why:** matches the baseline run and M2. **Tradeoff accepted:** no check that the bias results hold across list lengths (Coverage, ARP and Gini all depend strongly on K).

### D-22 · Gini definition — **Decided** (2026-11-04)
- **Choice:** Gini of item exposure (number of users' top-K lists containing the item) over **all 3,706 catalog items, zeros included**. Sorted-values formula `sum (2i-n-1) x_(i) / (n sum x)`. Range 0 to (n-1)/n.
- **Rejected:** over recommended items only. A model recommending the same 50 items to everyone would score a perfect 0, rewarding collapse, and the item set would change per configuration.
- **Tradeoff accepted:** overlaps with Coverage (low coverage raises Gini). With K=20 a good model still has a Gini well above 0, so read changes relative to the baseline.

### D-23 · ARP definition — **Decided** (2026-11-04)
- **Choice:** mean raw training degree of recommended items, **plus** the same statistic for each user's held-out test items as a reference (per user first, then averaged over users with held-out items).
- **Correction:** the ROADMAP listed "per user then averaged vs. pooled" as a choice. With exactly K items per user the two are identical for recommendations, so it is not a real choice there. It does matter for the held-out reference (users have different numbers of test items), which is why that one is per-user.
- **Options rejected:** raw degree only; degree divided by number of users.

### D-24 · Coverage denominator — **Decided** (2026-11-04)
- **Choice:** all 3,706 catalog items (consistent with D-04). **Rejected:** only items with at least one train edge, which would make the denominator vary by split and exclude the cold items Intervention C targets.

### D-25 · Recall and NDCG normalization — **Decided** (2026-11-04)
- **Recall:** hits / number of held-out items (D-18 confirmed). Keeps the existing baseline test Recall@20 of 0.2733 comparable and matches the LightGCN reference code. **Caveat for the report:** users with more than K held-out items cannot reach recall 1, which lowers absolute Recall equally for every configuration.
- **NDCG:** binary relevance, log2 discount, ideal list capped at min(K, number of held-out items), so a perfect ranking scores 1.
- Users with no held-out items are skipped for Recall and NDCG; Coverage, Gini and ARP use all users.

---

## M4 — Interventions

Standing rule from the user (2026-11-04): follow the proposal as written; deviate only where necessary. Where the proposal is silent, take the smallest change from the existing code.

### D-26 · Intervention A: degree-discounted aggregation — **Decided** (2026-11-04)
- **Choices:**
  1. **Where:** fold the discount into the adjacency, built once (`build_norm_adj(..., gamma)` returns `S @ A_hat`). `model.py` and the training loop are unchanged. The alternative, scaling inside `propagate`, has identical math for one gamma but touches the model and runs every step; it would only be needed for per-layer or learned gamma.
  2. **Direction and layers:** Eq. 1 literally: item-update direction only, at every layer. The propagation matrix is therefore **asymmetric** (user rows unscaled). Rejected: a symmetric version (also discount item->user), which is a different intervention from the proposed one; first-layer-only.
  3. **No re-normalization.** Eq. 1 has none. (Re-normalizing each item row by its own sum would cancel the discount exactly; only a global rescale would be meaningful, and the proposal has none.) **Consequence:** popular items' propagated embeddings shrink, so overall embedding scale changes with gamma, which interacts with the learning rate and the L2 term. Layer 0 is never discounted and is part of the layer mean, so the effect is partial.
  4. **Degree-0 items (P-02, resolved for A):** scale factor 0, consistent with D-07. It never affects the output, since a degree-0 item's row is already all zeros; the guard only avoids `0 * inf = NaN`.
  5. **Config:** `gamma` added to `TrainConfig` (default 0.0), so `run_baseline` gets `--gamma` for free.
- **Evaluation uses the same discounted adjacency as training** (the adjacency is part of the model's propagation).
- **Verified:** gamma=0 gives a bitwise-identical adjacency to the committed baseline (checked against the pre-change `adjacency.py` on a random 300x500 graph); hand-computed gamma=0.5 on the tiny graph; asymmetry; no NaN with a degree-0 item; training with gamma=0.5 differs from baseline and still learns.
- **Update (2026-11-04):** a 30-epoch training run on the planted test dataset with gamma=0, beta=0 gives bitwise-identical loss, L2 and Recall histories to the committed pre-intervention code (run from a `git archive` of HEAD). **Still not verified:** the real ML-1M baseline (Recall 0.2733) was not rerun on Kaggle.

### D-27 · Intervention B: popularity-aware loss re-weighting — **Decided** (2026-11-04)
- **Choices (all follow the proposal literally, per the standing rule):**
  1. **No weight normalization:** raw `w(i) = pop(i)^-beta`. **Flagged confound:** with raw weights the average weight falls well below 1 as beta rises (at beta=1 from 1 for rare items to about 1/2,700 for the most popular), shrinking the BPR term while the L2 term keeps its fixed lambda. Adam probably cancels a uniform loss rescale (untested), but the BPR/L2 balance shifts, so B's measured effect mixes "less popular-item gradient" with "relatively stronger regularization". Rejected option: rescale weights to mean 1 over train edges (a constant factor, keeps the balance, but deviates from Eq. 2); per-batch weight normalization.
  2. **L2 term is not weighted** (Eq. 2 touches only the BPR term; P-05 resolved). Popular items' BPR gradient shrinks while their regularization stays at baseline strength, an extra downward pull on them that is part of B's measured effect. Rejected: weighting the L2 term per sample.
  3. **Reduction:** weighted sum divided by batch size, like the baseline's mean. Necessary, not a free choice: the proposal's plain sum would scale BPR by the batch size relative to L2 and break beta=0 reproducing the baseline.
  4. **`pop(i)` = training degree**, same quantity as in A and ARP. Never 0 for a weighted item (a positive is a train edge), so the degree-0 question (P-02) does not arise for B; `pop_weights` clamps at 1 only to avoid inf for unused items.
  5. **Weight applies to the positive item only**; negatives are unweighted (as in Eq. 2). `beta` added to `TrainConfig` (default 0.0).
- **Implementation:** `pop_weights` computes per-item weights once from train degrees; `bpr_loss(pos, neg, weights)` multiplies each sample's term by its weight and takes the mean.
- **Consequence for logging:** the logged `bpr_loss` is the weighted loss, so it is not comparable across different beta values.
- **Property:** total gradient mass per item over an epoch is about `deg^(1-beta)`: proportional to popularity at beta=0, equal across items at beta=1.
- **Verified:** hand-computed weighted loss; unit weights bitwise equal to unweighted; gradient scales with the weight; weights decrease with popularity; beta=0 gives exactly 1; training with beta=0.5 differs from baseline and still learns; gamma=0, beta=0 reproduce committed baseline training bitwise (above).

### D-28 · Intervention C: content-based item embeddings — **Decided** (2026-11-04)
- **Eq. 3:** `e_i^(0) = e_i^ID + f_theta(x_i)` for **every** item (not the threshold idea in P-01). `f_theta` is trained jointly with everything else by the same BPR loss, in the same optimizer step; there is no pretraining and no extra loss.
- **Choices:**
  1. **MLP:** one hidden layer, ReLU, hidden width 64, output = embed_dim, no dropout ("lightweight MLP"; the proposal gives no details). 5,440 parameters at embed_dim 64 and 19 input features. Rejected: linear-only (not an MLP), deeper or dropout.
  2. **Genres:** multi-hot over the 18 ML-1M genres. **Interpretation flagged:** the proposal says "one-hot", but real movies average 1.67 genres (max 6), so a strict one-hot would drop information; multi-hot is the standard reading of one-hot encoding for multi-label data. Rejected: one-hot of the first listed genre only.
  3. **Year:** parsed from the last "(YYYY)" in the title, min-max scaled to [0, 1] over the catalog (range 1919-2000). Missing years would be filled with the mean year; **in ML-1M there are none** (0 of 3,883 titles).
  4. **Optimization:** the MLP's weights join the existing Adam optimizer at the same learning rate, with no extra regularization. The L2 term (D-15) still penalizes only the ID embeddings. Rejected: separate learning rate, weight decay on the MLP, L2 on the combined embedding.
  5. **Init scale:** PyTorch default init for the MLP (smallest change). **Measured on real features:** the initial MLP output has std about 0.12 versus 0.01 for the ID embeddings (D-14), so early in training the content part dominates every item's layer-0 embedding, not only cold items. Not corrected, to stay with the default. If C looks odd in M5, shrinking the last layer's init is the first thing to try.
  6. **Off switch:** `use_content` (default False). When off, no MLP is created, so the model and its random init are the baseline's, bit for bit.
- **Mechanics:** the item features are a fixed non-persistent buffer (not saved in `model.pt`, rebuilt from `items.csv` at load time); the MLP is built after the embeddings so the ID init is unchanged by C. Features are built in `data/features.py` from `items.csv`, catalog items only, ordered by item index.
- **CLI fix:** `run_baseline` flags for boolean fields now parse "True"/"False" properly (`type=bool` would treat "False" as True).
- **Cold items:** a zero-train-degree item has all-zero propagated layers, so its final embedding is `e^(0) / (K+1)`; with C that is dominated by `f_theta(x)`, which other items trained. Whether this helps is **not evaluated here**: there are only 22-31 such items per split, and the head/mid/tail breakdown is M6.
- **Verified:** feature parsing hand-computed on a tiny table and run on the real items table (3,706 x 19, no NaN); layer-0 is ID + MLP; gradients reach both the MLP and the ID embeddings; MLP weights move during training; training with C differs from baseline and still learns the planted structure; C-off model and training are bitwise identical to the committed baseline (checked against a `git archive` of HEAD with all of A, B, C off); CLI bool parsing; end-to-end CLI run with C on.
- **Not verified:** that C improves anything on real ML-1M, and no full real-data training run has been done locally (about 340 s/epoch on CPU).

---

## M5 — Experiment grid

### D-29 · Grid shape — **Decided** (2026-11-05); stopping rule and dial selection still **Open**
- **Constraint:** 12 days in total, and a Kaggle GPU quota believed to be about 30 GPU-hours a week (unverified), at about 21.6 s/epoch (P-06). The proposal's "a few hours total" does not hold: one run is about 1.2 h at 200 epochs.
- **Choice (option Y):**
  - **Stage 1** (split 0, model seed 0, selection on validation only): baseline, C alone, A alone at gamma in {0.25, 0.5, 0.75}, B alone at beta in {0.25, 0.5, 0.75}. 8 runs.
  - **Stage 2:** all 8 on/off combinations of A, B, C (baseline, A, B, C, AB, AC, BC, ABC) at the chosen gamma\* and beta\*, on **2 repeats** (repeat r = saved split r and model seed r, r in {0, 1}). The split-0 runs of baseline, A, B and C come from stage 1, so 12 are new.
  - **Total: 20 runs** (about 24 GPU-hours at 200 epochs, 18 h at 150).
- **Dropped:** gamma = 1 and beta = 1, so the tested range is [0.25, 0.75] and not the proposal's full [0, 1]; the report must say so. Rationale: beta = 1 is where un-normalized weights are most confounded with the L2 term (D-27) and gamma = 1 is the likeliest to be degenerate; they only save 2 runs, so they can be added back at the end if quota remains. Also dropped: the 5-value sweeps, a joint gamma x beta cross, and the third repeat.
- **What the repeats are:** independent models, each on its own random 80/10/10 split (D-02), reporting mean and spread. Not cross-validation, no ensembling. Split and seed vary together, so their effects are not separated. All 8 configurations share the same splits, enabling same-split comparisons.
- **Statistical caveat to state in the report:** a spread from 2 points is unreliable, so report both values or the range. A third split (5 more runs: baseline, A, B, C, ABC) can be added last if the 2-split gap looks too close to the effect sizes.
- **Second caveat:** gamma\* and beta\* are chosen on split 0's validation set, so split 0 is slightly matched to the tuned configurations. Validation and test are different edges, so this is not leakage.
- **Decided (2026-11-05):**
  - **Epoch budget:** fixed 200 epochs for every run, evaluated every 10, best-validation checkpoint kept, no early stopping (identical for all configs, D-12). Reason: the baseline did not peak until epoch 290, so a shorter budget is hard to justify; 200 epochs reaches about 97% of the baseline's best validation Recall (0.2416 vs 0.2492). **Risk accepted:** a configuration that converges more slowly is cut off and may be penalized.
  - **Saved per run:** one metrics row (validation and test, all five metrics) plus the top-20 recommendation lists (about 0.5 MB), so M6's head/mid/tail breakdown needs no further GPU time.
  - **Persistence across Kaggle sessions:** `results/results.jsonl` and the recommendation files live in the repo; after each session the user downloads the output and commits it; the next session clones the repo and skips finished runs.
  - **Choosing gamma\* and beta\* (2026-11-05):** among the stage 1 dial values (0.25, 0.5, 0.75), pick the one with the **lowest validation Gini whose validation Recall@20 is within 5% (relative) of the stage 1 baseline's**. A and B are chosen separately, each against the same baseline run. **Fallback:** if no value is within 5%, pick 0.25 and print a warning. A `select_dials` step prints the choice and writes `results/dials.json` for the user to review and commit before stage 2.
    - **Rejected:** highest validation Recall (almost always the weakest dial, so the combined method would barely differ from the baseline and the hypothesis could not be tested); fixed 0.5 with the sweep only as the trade-off curve (no justification beyond being the middle).
    - **Caveats:** the 5% tolerance is a judgment call that cannot be derived; stage 1 is one run per dial value, so the Recall comparison is noisy at the margin; A and B together may lose more than 5%, which stage 2 will show. **Sensitivity:** every stage 1 run is logged, so the report can show what a 3% or 10% tolerance would have picked.
  - **Runs per Kaggle session (2026-11-05):** one. The runner (`run_grid`) takes `--max_runs N` (default 1) and does the next N unfinished runs in a fixed order; `--run_id` runs one named run; `--max_hours` stops before a run that would not fit. Reason: a crashed or timed-out session then loses at most one run (about 1.2 h), which matters because it is **unverified** whether Kaggle keeps the output of a session that dies. **Cost accepted:** about 20 manual download/commit cycles and a few minutes of setup per session (unmeasured). One notebook, `notebooks/grid_run.ipynb`, is saved as a new version each time, not 20 separate notebooks.
- **Implementation notes:** a run's identity is `g<gamma>_b<beta>_c<0|1>_s<split>` (model seed = split), so runs shared by both stages (split 0 of baseline, A, B, C) are trained once. Rows whose epoch budget is not 200 never count as finished and are ignored by `select_dials`, so a smoke run cannot pollute the grid. Each row holds validation and test values of all five metrics plus the held-out popularity reference, the config, the git commit and the validation history; the test top-20 lists go to `recs/<key>.npz` as int16.

---

## Infrastructure

### D-11 · Code organization and where to run — **Decided** (2026-10-16)
- **Choice:** all real code in `.py` files in the repo; two thin Kaggle notebooks (an experiment notebook and an analysis notebook). Runs happen on Kaggle GPUs, not the local machine.
- **Options:** package + thin notebooks; everything in notebooks; scripts only.
- **Why:** small, reviewable diffs, tests run on CPU, reproducible runs from a config and seed, and each plot is one notebook cell.
- **Tradeoffs:** a little wiring to get the repo onto Kaggle (GitHub clone or private Kaggle Dataset; internet must be enabled for cloning); Kaggle has session-time and weekly GPU quotas (verify current limits).
- **Still open:** how the code gets onto Kaggle (is the repo on GitHub?), and the experiment-tracking tool (M5).

---

## Parked for later milestones

### P-01 · Threshold-based Intervention C (M4c)
- Idea from the user: apply the content initialization only to items below a rating-count threshold, instead of all items.
- **Flag:** Eq. 3 in the proposal applies `f_θ(x_i)` to every item, so this deviates from the proposal. Decide knowingly in M4c.

### P-02 · Degree-0 handling in Interventions A and B (M4)
- `|N(i)|^-γ` and `pop(i)^-β` are undefined at degree 0. Needs a rule when each intervention is designed (see D-07).

### P-03 · M2 sanity-check reference
- The LightGCN paper did not use ML-1M, so no directly comparable published Recall@20 exists. Define the reference in M2.

### P-05 · Does Intervention B's weight apply to the L2 term? (M4b)
- B reweights each sample's BPR loss by `pop(i)^-beta`. The L2 term (D-15) is per-sample too; decide whether to weight it as well. Affects how strongly popular vs. tail embeddings are regularized.

### P-06 · Grid cost at the measured training speed (M5)
- At ~21 s/epoch, one run is likely 1-3 hours (unknown until the first full run finishes). 8 on/off combinations plus a gamma/beta sweep, times 3-5 splits, times 3-5 seeds could be hundreds of runs. The planning assumption "each run takes minutes on a T4" does not hold unless runs converge much faster than expected. Decide in M5: epoch budget, stopping rule, which parts of the grid to run, and whether to speed up training.

### P-04 · Experiment tracking (M5)
- Per-run results rows (config, seed, split, git commit, metrics) plus Kaggle "Save Version". Whether to add a tool such as Weights & Biases is an M5 decision.
