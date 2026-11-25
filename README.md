# Combating Popularity Bias in LightGCN

ECE 592 / CSC 591 course project (NC State). A from-scratch PyTorch implementation of
[LightGCN](https://arxiv.org/abs/2002.02126) on MovieLens-1M, with three interventions against popularity bias that
can each be switched on or off:

| | Intervention | Where it acts | Switch |
|---|---|---|---|
| **A** | Degree-discounted aggregation: item updates scaled by `\|N(i)\|^-γ` | propagation | `gamma` (0 = off) |
| **B** | Popularity-aware loss re-weighting: BPR term times `pop(i)^-β` | training loss | `beta` (0 = off) |
| **C** | Content-based item embeddings: `e_i^(0) = e_i^ID + f_θ(genres, year)` | initialization | `use_content` |

## Results (test set, K = 20, mean of 3 splits)

| Config | Recall@20 | NDCG@20 | Coverage | Gini | ARP |
|---|---|---|---|---|---|
| Baseline | 0.2681 | 0.2852 | 0.588 | 0.848 | 1006 |
| A | 0.2573 | 0.2758 | 0.598 | 0.851 | 1013 |
| B | 0.2582 | 0.2768 | 0.649 | 0.831 | 986 |
| C | 0.2622 | 0.2801 | 0.605 | 0.840 | 987 |
| AB | 0.2412 | 0.2568 | 0.671 | 0.815 | 930 |
| AC | 0.2485 | 0.2665 | 0.605 | 0.840 | 978 |
| BC | 0.2505 | 0.2685 | 0.662 | 0.824 | 963 |
| ABC | 0.2267 | 0.2443 | 0.678 | 0.809 | 896 |

Higher is better for Recall, NDCG and Coverage; lower is better for Gini and ARP. γ = β = 0.25. No configuration beats
the baseline on both accuracy and popularity bias. K = 5 and K = 10 results are in `results/metrics_at_k.csv`.

## Which code was developed for this project

All code in `src/`, `scripts/`, `tests/` and `notebooks/` was developed for this project. No existing LightGCN
implementation or recommender-system library is used.

| Developed in this project | Where |
|---|---|
| ML-1M parsing, implicit-feedback edges, per-user 80/10/10 splits | `data/preprocess.py`, `data/download.py` |
| Normalized bipartite adjacency, and its degree discount (**Intervention A**) | `data/adjacency.py` |
| LightGCN model, and the content MLP (**Intervention C**) | `model.py`, `data/features.py` |
| BPR loss, popularity weights (**Intervention B**), batch L2 term | `losses.py` |
| Uniform negative sampler, training loop | `sampler.py`, `train.py` |
| Full-ranking evaluation: Recall, NDCG, Coverage, ARP, Gini | `evaluate.py`, `metrics.py`, `metrics_at_k.py` |
| Experiment grid, dial selection, figures and tables | `run_grid.py`, `select_dials.py`, `scripts/` |
| Unit and end-to-end tests | `tests/` |

**Third-party libraries:** PyTorch (tensors, sparse matrix multiply, autograd, `nn.Embedding`/`nn.Linear`, Adam),
NumPy, pandas, requests, matplotlib (figures) and pytest (tests).

**References used:** the LightGCN paper and the authors' [official repository](https://github.com/kuandeng/LightGCN)
were consulted for hyperparameters, embedding initialization and the form of the L2 term.

**AI assistance:** design decisions were made by the project author. An AI coding assistant (Claude Code) was used to
propose options, to write code and tests for the chosen options, and to draft documentation. The working rules are in
`CLAUDE.md`, and each design decision with its alternatives is logged in `DESIGN.md`.

## Setup

Python 3.10 or newer.

```bash
git clone https://github.com/rithikkulkarni/longtail-lightgcn.git
cd longtail-lightgcn
pip install -e ".[dev,figures]"
```

A GPU is only needed for training. Tests, metrics and figures run on CPU.

## Running the code

Run every command from the repository root.

### 1. Prepare the data

```bash
python scripts/prepare_data.py          # downloads ML-1M, writes data/processed/split_0..4.npz and items.csv
```

The splits are seeded, so every machine gets the same ones. `data/` is not committed.

### 2. Run the tests

```bash
python -m pytest
```

### 3. Reproduce the report's numbers and figures (no training, no GPU)

The finished runs are committed in `results/`, so this needs only step 1.

```bash
python -m longtail_lightgcn.metrics_at_k        # all metrics at K = 5, 10, 20 -> results/metrics_at_k.csv
python scripts/make_figures.py                  # -> report/figures/pareto_k*.pdf and report/tables/*.tex
python scripts/make_method_figure.py            # -> report/figures/method_overview.pdf
```

`metrics_at_k` first recomputes the K = 20 metrics from the saved recommendation lists and stops if they do not match
what training recorded.

### 4. Train a single model

Training takes about 1.2 hours per 200-epoch run on a Kaggle GPU, and about 6 minutes **per epoch** on a CPU.

```bash
# baseline LightGCN
python -m longtail_lightgcn.run_baseline --split_seed 0 --seed 0 --out_dir results/my_baseline

# all three interventions
python -m longtail_lightgcn.run_baseline --gamma 0.25 --beta 0.25 --use_content True --out_dir results/my_abc
```

Every field of `TrainConfig` (`config.py`) is a command-line flag. This script's defaults are up to 1000 epochs with
early stopping. The experiment grid instead uses a fixed 200 epochs, evaluated every 10; to match it, add
`--max_epochs 200 --eval_every 10 --patience 1000000`.

### 5. Re-run the experiment grid

`run_grid` skips every run already listed in `<results_dir>/results.jsonl`. To retrain from scratch, use a **new**
results folder.

```bash
OUT=results_new

# stage 1: dial sweep on split 0 (10 runs). Repeat until it reports the stage is complete.
python -m longtail_lightgcn.run_grid --stage 1 --results_dir $OUT --max_runs 10

# choose gamma* and beta* from validation results -> $OUT/dials.json
python -m longtail_lightgcn.select_dials --results_dir $OUT

# stage 2: all 8 combinations on splits 0, 1, 2 (20 new runs)
python -m longtail_lightgcn.run_grid --stage 2 --results_dir $OUT --max_runs 24

# show what is finished and what is pending
python -m longtail_lightgcn.run_grid --stage 2 --results_dir $OUT --list
```

Useful flags: `--run_id KEY` (one specific run), `--max_hours H` (do not start a run that would not fit),
`--device cpu|cuda`.

### On Kaggle

`notebooks/grid_run.ipynb` clones this repository, builds the data and runs the next unfinished grid runs.
`notebooks/baseline_run.ipynb` does the same for a single baseline run. Turn on Internet and a GPU.

## Repository layout

```
src/longtail_lightgcn/   model, loss, sampler, training, metrics, experiment runners
  data/                  download, preprocessing, adjacency, item features
scripts/                 data preparation and report figures/tables
tests/                   unit and end-to-end tests
notebooks/               thin Kaggle drivers
results/                 results.jsonl, dials.json, metrics_at_k.csv, recs/<run>.npz
DESIGN.md                design decision log          ROADMAP.md   milestones
```

### Result files

- `results/results.jsonl`: one line per run: configuration, git commit, best epoch, validation and test metrics at
  K = 20, and the validation history. A run is named `g<gamma>_b<beta>_c<0|1>_s<split>`; the model seed equals the split.
- `results/recs/<run>.npz`: each user's test top-20 items, best first (shape 6040 × 20).
- `results/dials.json`: the chosen γ\* and β\* with the table behind the choice.
- `results/metrics_at_k.csv`: one row per run and K.

## Notes on reproducibility

- Splits, model initialization and negative sampling are seeded. GPU arithmetic is not guaranteed to be bit-for-bit
  repeatable, so retrained numbers may differ slightly.
- Hyperparameters come from the LightGCN authors' settings for other datasets; they were not tuned on ML-1M.
- Stage 1 also includes γ = 1 and β = 1. They were added after the dial values had been chosen and are not selection
  candidates.

## File and function reference

<!-- generated from the docstrings -->

### `src/longtail_lightgcn/config.py`

Hyperparameters, defaulting to the LightGCN authors' README settings (DESIGN.md D-12, D-14, D-19).

- **class `TrainConfig`**: All settings of one training run. The defaults are the baseline (interventions A, B, C off).

### `src/longtail_lightgcn/data/download.py`

Download and unpack MovieLens-1M into a raw data directory.

- `download_ml1m`: Fetch ML-1M into `raw_dir` and return the folder holding the .dat files.

### `src/longtail_lightgcn/data/preprocess.py`

Parse ML-1M, build the implicit-feedback edge list, split it, and save results.

- `load_ratings`: Read ratings.dat. Rating and timestamp are dropped: every rating is an edge (D-01).
- `load_movies`: Read movies.dat (title and raw pipe-separated genre string, no feature engineering).
- `build_edges`: Remap raw IDs to contiguous indices and return the edge list.
- `holdout_size`: 10% of n rounded half-up (D-09). Integer form of floor(0.1*n + 0.5), avoiding float error.
- `make_split`: Per-user random 80/10/10 split (D-02).
- `save_split`: Write one split's edge arrays and the user/item counts to an .npz file.
- `load_split`: Return (split_dict, n_users, n_items).
- `save_id_maps`: Write the arrays that map model indices back to the original MovieLens user and movie IDs.
- `save_items_csv`: Write metadata for ALL movies in movies.dat, flagging which ones are in the catalog (D-04).
- `degree_stats`: Counts used to decide whether the zero-train-degree rule (D-07) needs revisiting.

### `src/longtail_lightgcn/data/adjacency.py`

Normalized bipartite adjacency for LightGCN (DESIGN.md D-06, D-07).

- `build_norm_adj`: Return D^-1/2 A D^-1/2 as a coalesced COO tensor of shape (U+I, U+I).

### `src/longtail_lightgcn/data/features.py`

Item content features for Intervention C (DESIGN.md D-28): multi-hot genres + normalized release year.

- `parse_year`: Release year from a title ending in "(YYYY)"; NaN when there is none.
- `build_item_features`: Feature matrix for the catalog items, rows ordered by item index.
- `load_item_features`: Read items.csv (written by scripts/prepare_data.py) and return the feature matrix.

### `src/longtail_lightgcn/model.py`

LightGCN: parameter-free propagation over the normalized bipartite graph (He et al., 2020).

- **class `LightGCN`**: Learnable parameters are only the layer-0 (ID) embeddings of users and items.
  - `__init__`: Create the ID embedding tables and, if item_features is given, the content MLP of Intervention C.
  - `ego_embeddings`: Layer-0 embeddings stacked as (U+I, d); users first, items after, matching the adjacency.
  - `propagate`: Run K propagation layers and return (user_final, item_final).
  - `score`: Dot-product score for each (user, item) pair in the batch.

### `src/longtail_lightgcn/losses.py`

BPR loss and the batch-level L2 term (DESIGN.md D-15), matching the authors' official code.

- `pop_weights`: Intervention B (D-27): w(i) = pop(i)^-beta per item, pop = training degree, shape (n_items,).
- `bpr_loss`: Mean over the batch of w * -log sigmoid(pos - neg); w = 1 when `weights` is None.
- `l2_reg`: 0.5 * (sum of squared layer-0 embeddings of the batch's users, positives, negatives) / batch size.

### `src/longtail_lightgcn/sampler.py`

Uniform negative sampling with rejection of train positives (DESIGN.md D-16).

- **class `NegativeSampler`**: Draw one negative item per (user, positive) pair.
  - `__init__`: Index the training edges for fast membership tests and seed the random generator.
  - `_is_train_edge`: Boolean array: whether each (user, item) pair is a training edge (binary search).
  - `sample`: Return one negative item for each entry of `users`.

### `src/longtail_lightgcn/train.py`

Training loop for LightGCN (baseline and interventions A, B, C): BPR loss, uniform negative sampling, and selection of the checkpoint with the best validation Recall@K (D-12, D-16).

- **class `TrainResult`**: Outcome of a training run: the best-validation model, its adjacency matrix and the validation history.
- `evaluate`: Recall@K on validation (exclude train) or test (exclude train + val).
- `train`: `item_features` (n_items, F) is required when cfg.use_content is on (Intervention C) and ignored otherwise.

### `src/longtail_lightgcn/evaluate.py`

Full-ranking Recall@K used for validation during training (DESIGN.md D-13, D-18). All five report metrics are in metrics.py.

- `_edge_mask`: Dense (n_users, n_items) boolean matrix, True where an edge exists in any of `pairs`.
- `recall_at_k`: Full-ranking Recall@K, averaged over users that have at least one target item.

### `src/longtail_lightgcn/metrics.py`

Ranking and popularity-bias metrics (DESIGN.md D-21 to D-25).

- `recommend_top_k`: Top-K item indices per user over the full catalog, with already-seen items masked out.
- `target_mask`: Dense bool (n_users, n_items) matrix of held-out edges, from a list of (users, items) arrays.
- `recall_at_k`: Mean over users with >= 1 held-out item of hits / number of held-out items (D-18).
- `ndcg_at_k`: Binary-relevance NDCG@K with a log2 discount, averaged over users with >= 1 held-out item.
- `item_exposure`: How many users' lists contain each item, shape (n_items,); zero for never-recommended items.
- `coverage`: Share of the whole catalog (n_items, D-04) recommended to at least one user.
- `gini`: Gini coefficient of item exposure over ALL catalog items, zeros included (D-22).
- `average_recommendation_popularity`: Mean training degree of recommended items (D-23).
- `target_popularity`: Reference for ARP: mean training degree of each user's held-out items, averaged over users.
- `all_metrics`: All five proposal metrics plus the held-out popularity reference (D-23), as plain Python floats.

### `src/longtail_lightgcn/metrics_at_k.py`

Recompute every test metric at K = 5, 10 and 20 from the saved top-20 lists (no training, no GPU).

- `metrics_at_ks`: {K: {metric: value}} for the first K columns of a best-first top-N list (N >= max(ks)).
- `check_against_stored`: Largest absolute difference between the recomputed K=20 metrics and the stored ones; raises above tol.
- `build_table`: Metrics at every K for every finished run, checked at K = 20 against results.jsonl. Returns (table, largest K=20 difference, number of runs).
- `main`: Command-line entry point: build the table and write it to metrics_at_k.csv.

### `src/longtail_lightgcn/run_baseline.py`

Train baseline LightGCN on one saved split and write results to disk.

- `git_commit`: Hash of the code that produced the run, for traceability (empty if not in a git repo).
- `_str_to_bool`: argparse's type=bool treats any non-empty string (even "False") as True, so parse explicitly.
- `build_parser`: Command-line parser with one flag for every TrainConfig field.
- `main`: Train one configuration on one split, evaluate the test set once, and save results.json and model.pt.

### `src/longtail_lightgcn/run_grid.py`

Experiment grid runner (DESIGN.md D-29).

- **class `RunSpec`**: One run of the experiment grid: which interventions are on, their strengths, and the split.
  - `key`: Unique run identifier, e.g. g0.25_b0_c1_s2 (gamma, beta, content flag, split).
  - `label`: Which interventions are on: 'baseline', 'A', 'B', 'C', 'AB', ..., 'ABC'.
- `stage1_specs`: Stage 1 runs on split 0: baseline, C, the A and B dial sweeps, then the gamma = 1 and beta = 1 end-points.
- `stage2_specs`: Stage 2 runs: the 8 on/off combinations at the given gamma and beta, one split at a time.
- `read_rows`: Read results.jsonl into a list of dicts; unreadable lines are skipped with a warning.
- `append_row`: Append one JSON line and force it to disk, so a finished run survives a crash right after.
- `done_keys`: Keys of finished runs at this epoch budget (a smoke run with a different budget never counts).
- `load_dials`: Read the chosen gamma* and beta* from dials.json; exit with instructions if it is missing.
- `run_one`: Train one run, compute validation and test metrics from the best checkpoint, save its test top-K lists, and return the results row.
- `main`: Command-line entry point: list, or train, the next unfinished runs of a stage.

### `src/longtail_lightgcn/select_dials.py`

Choose gamma* and beta* from the stage 1 runs (DESIGN.md D-29).

- `select_dials`: Apply the selection rule to the stage 1 rows, using validation metrics only. Returns the chosen gamma and beta with the table behind the choice.
- `main`: Command-line entry point: print the selection table and write dials.json.

### `scripts/make_figures.py`

Build the report's figures and LaTeX tables from the saved results (no torch needed).

- `load_stage2`: Stage-2 rows (the 8 on/off configs at gamma*, beta*) restricted to splits where all 8 finished.
- `summarize`: Per-config mean and std (over splits) of every metric at K.
- `percent_changes`: Per split: 100 * (config - baseline) / baseline for Recall and Gini; then mean/std over splits.
- `pareto_frontier`: Labels not dominated by any other point (lower Gini AND higher Recall is better), in Gini order.
- `num`: Format a number with d decimals.
- `pm`: LaTeX cell 'mean +/- std', or the mean alone when std is hidden or undefined.
- `signed`: Format a percent change with an explicit plus or minus sign.
- `write_table`: Write a LaTeX table to disk, creating the folder if needed.
- `results_table`: LaTeX table of all five metrics for the 8 configurations at one K; best value per column in bold.
- `delta_table`: LaTeX table of percent change in Recall and Gini against the baseline at one K.
- `dial_sweep_table`: LaTeX table of the stage 1 dial sweep (validation, split 0); chosen dials in bold.
- `label_offset`: Label above-right by default; below-right when another mean sits close by and is at least as high.
- `pareto_figure`: Plot Recall@K against Gini (means and per-split points) with the Pareto frontier.
- `main`: Command-line entry point: write every result table and figure.

### `scripts/make_method_figure.py`

Draw the method overview figure: the LightGCN pipeline and where interventions A, B and C act.

- `box`: Draw one titled stage box; added=True uses the intervention colour.
- `arrow`: Draw an arrow between two points; added=True uses the intervention colour.
- `main`: Command-line entry point: draw and save the method overview figure.

### `scripts/prepare_data.py`

Download ML-1M, build the edge list, and write N splits plus metadata to data/processed/.

- `main`: Download ML-1M, build the edge list, and save the splits and item metadata.

### `tests/`

| File | What it checks |
|---|---|
| `test_adjacency.py` | Tests for the normalized adjacency, including the degree discount of Intervention A. |
| `test_content.py` | Intervention C: content-based item embeddings, e_i^(0) = e_i^ID + f_theta(x_i) (DESIGN.md D-28). |
| `test_evaluate.py` | Tests for full-ranking Recall@K and its train/validation masking. |
| `test_features.py` | Tests for the item content features (genres and release year). |
| `test_losses.py` | Tests for the BPR loss, popularity weights (Intervention B) and the L2 term. |
| `test_metrics.py` | Tests for Recall, NDCG, Coverage, ARP and Gini on hand-computed examples. |
| `test_metrics_at_k.py` | Tests for metrics_at_k.py: the first-K-columns trick must equal asking for the top-K list directly. |
| `test_model.py` | Tests for LightGCN propagation and scoring on a tiny hand-computed graph. |
| `test_preprocess.py` | Tests for ID remapping, the per-user split and saved files. |
| `test_run_baseline.py` | End-to-end test of the single-run command-line script. |
| `test_run_grid.py` | Tests for the experiment grid runner and the dial selection rule. |
| `test_sampler.py` | Tests for uniform negative sampling with rejection of training positives. |
| `test_train.py` | Tests for the training loop and checkpoint selection. |
