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

### P-04 · Experiment tracking (M5)
- Per-run results rows (config, seed, split, git commit, metrics) plus Kaggle "Save Version". Whether to add a tool such as Weights & Biases is an M5 decision.
