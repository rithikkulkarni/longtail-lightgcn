# Presentation Notes — Combating Popularity Bias in LightGCN

Slide-by-slide plan based on the project proposal, DESIGN.md, `results/dials.json` and `results/results.jsonl`. I haven't seen the professor's instructions, so the time limit and slide count are unknown. The plan below is about 14 slides.

## Status of the runs

16 of 20 runs are in `results.jsonl`. Four are still outstanding: AB, AC, BC and ABC on split 1. All the other configs have both splits.

- **Results table (slide 11):** use split 0, where all 8 configs are complete. Footnote it "Split 0 shown; split 1 in progress". If the remaining runs finish before you present, add a second column for split 1, or report the mean and range across both. Show both values, because two points make a poor spread estimate (D-29).
- **Discussion (slide 12):** the claim about ABC is the one most likely to change. ABC and AB have only a single split so far, and they carry the headline conclusion. Write "on split 0, ABC lowers Gini most, at the cost of a 15% Recall drop". Do not write "ABC does not beat B alone" as a settled finding.
- **Individual interventions:** A, B and C each have both splits, so those conclusions are safer. On split 1, A (Gini 0.852 vs baseline 0.847) and C (0.839 vs 0.847) point the same way as on split 0. B (0.832) is the exception. I haven't checked how its Recall and Gini compare with the baseline across the two splits, so check that before claiming B's result is stable.
- When the last four runs land, re-read `results.jsonl` and update the table and the discussion numbers.

## The negative-sample question

The understanding is mostly right, but two points need fixing before presenting.

**What's right**
- The negatives are all unwatched movies (D-16). An unwatched movie is not a disliked movie, so some negatives are movies the user would have liked. This is the "false negative" problem. Implicit feedback is "missing not at random": a missing edge could mean "not interested" or "never saw it".
- Filtering to liked movies does cost data. The D-01 estimate is about 575K edges at rating ≥ 4, versus about 1.0M. Thresholding also changes the degree distribution, which changes what "popular" means for A, B and Gini.

**What to fix**
1. **The dataset does contain dislike signals.** ML-1M has explicit 1–5 ratings, and the pipeline binarized them away (D-01: every rating ≥ 1 is an edge). So a rating of 1 or 2 is a recorded dislike that the pipeline currently treats as a positive. The accurate framing is: "the data has explicit dislikes, but only about 16% of ratings are ≤ 2 (recollection of the ML-1M rating distribution, so verify it). That is too few to serve as the negative set, and they are not a random sample of what users dislike."
2. **Separate three questions.** They are easy to blur, and an interviewer may pull them apart.
   - Which edges count as positives: the threshold choice.
   - Where negatives come from: unwatched items (current choice) versus low-rated items.
   - How negatives are sampled: uniform (current choice) versus popularity-weighted.

   Lowering the positive threshold shrinks the positives. Using low ratings as negatives is a separate change.
3. **Link it to the topic.** Uniform negatives mean a popular item is pushed up about `deg(i)` times per epoch but pushed down only about 216 times (D-16). This bias exists in the baseline before any intervention. It is the strongest bridge from "the negatives are shaky" to "this causes popularity bias". It also affects evaluation, because held-out liked items that were never watched count as misses.

**Options to name as future work**
- Positives at rating ≥ 4.
- Low ratings as hard negatives.
- Popularity-aware negative sampling. This was rejected for the baseline in D-16 because it acts as a hidden intervention, which is a good defensible point.

## Slide-by-slide

### 1. Title
- Title, name, course.
- One-line thesis: "Does fixing popularity bias at three different pipeline stages work, and what does it cost in accuracy?"

### 2. NGCF → LightGCN
- Put on the slide:
  - NGCF's per-layer feature transform plus nonlinearity.
  - Why those are dead weight when nodes have only ID embeddings and no content features.
  - LightGCN keeps only normalized neighbor aggregation and a mean over layers (layer 0 included).
  - The ablation result and the "training difficulty, not overfitting" point.
- Visual: side-by-side layer diagrams, with the removed parts crossed out.
- Include the LightGCN propagation equation `Â = D^-1/2 A D^-1/2`.

### 3. The remaining issue: popularity bias
- Define it: a few items get most of the recommendations.
- Show the long-tail interaction curve from the ML-1M data (items sorted by train degree). Numbers: 3,706 items and about 1M edges, with some items having very few edges.
- Mention two sources:
  - Graph convolution: popular items aggregate more messages.
  - The training objective: uniform negatives push popular items up far more than down (D-16).
- Nuance for Q&A: symmetric normalization already partly dampens high-degree nodes per edge. So the proposal's "reinforced every layer" claim is directionally right but not the whole story.

### 4. Previous attempts
- Organize by where in the pipeline they act. This frames the three interventions on the next slides.
  - **Data and sampling:** popularity-based negative sampling.
  - **Training objective:** inverse-propensity weighting, popularity regularizers, causal approaches such as PD/PDA, MACR and DICE.
  - **Post-processing:** re-ranking.
  - **Cold-start and content:** content-aware models such as DropoutNet.
- Cite Klimashevskaia et al. (2024) as the survey and Lee et al. (2024) for LightGCN's inconsistencies.
- Flag: the specific method names and years are from memory. Check them before putting them on a slide.

### 5. The negative-sample problem
- Left side: what the data gives you.
  - Explicit ratings 1–5, binarized to "watched".
  - Unwatched is not the same as disliked.
- Middle: the three separate questions from above.
- Right side: tradeoffs.
  - Rating ≥ 4 loses about 42% of edges and reshapes popularity.
  - Low-rating negatives are few and not random.
  - The current choice is "all ratings as positives, uniform unwatched negatives", chosen because the study is about exposure, not preference.
- End with the BPR up/down asymmetry as a source of bias.

### 6. Setup
- Quick facts:
  - 6,040 users, 3,706 items, about 1M edges.
  - Per-user 80/10/10 split.
  - Full-ranking evaluation at K = 20.
  - Dimension 64, 3 layers, 200 epochs, from-scratch implementation.
- Metrics, one line each: Recall, NDCG, Coverage, ARP, Gini. Say that Gini is over all 3,706 items with zeros included (D-22).
- Keep it to one slide. It's the "can I trust this?" slide.

### 7. Intervention A: degree-discounted aggregation
- Equation 1: `e_i^(k+1) = Σ_u (1/√(|N(u)||N(i)|)) · |N(i)|^-γ · e_u^(k)`.
- Why: shrink popular items' updates.
- How it's implemented: a row scaling `S·Â` built once, which makes the matrix asymmetric.
- Be ready to defend that layer 0 is never discounted, and that there is no re-normalization (D-26).

### 8. Intervention B: popularity-aware loss re-weighting
- Equation 2: `w(i) = pop(i)^-β` on the positive item's BPR term.
- Why: equalize gradient mass per item. It goes from proportional to `deg` at β = 0 to equal across items at β = 1.
- Be ready for the flagged confound: raw weights shrink the BPR term while L2 stays fixed (D-27).

### 9. Intervention C: content-based initialization
- Equation 3: `e^(0) = e^ID + f_θ(x)`.
- `x` is an 18-genre multi-hot vector plus a scaled year, giving 19 features. `f_θ` is a one-hidden-layer MLP of width 64.
- Why: cold items get a meaningful position in the vector space.
- Flag: the proposal says one-hot, and the implementation uses multi-hot because movies average 1.67 genres.

### 10. Experimental design
- Two-stage grid:
  - Stage 1 swept the dials on validation (8 runs).
  - Stage 2 ran all 8 on/off combinations at γ* = 0.25 and β* = 0.25.
- Selection rule: lowest validation Gini within 5% of baseline Recall.
- Be upfront that the tested range is [0.25, 0.75], not the full [0, 1], and that only 2 splits were used because each run takes about 1.2 h.

### 11. Results
Test set, split 0, 200 epochs:

| Config | Recall@20 | NDCG@20 | Coverage | Gini | ARP |
|---|---|---|---|---|---|
| Baseline | .2677 | .2853 | .581 | .851 | 1014 |
| A | .2569 | .2756 | .595 | .851 | 1008 |
| B | .2591 | .2771 | .648 | .830 | 981 |
| C | .2630 | .2811 | .603 | .842 | 989 |
| AB | .2421 | .2575 | .671 | .813 | 921 |
| AC | .2494 | .2678 | .603 | .840 | 973 |
| BC | .2514 | .2702 | .659 | .826 | 968 |
| ABC | .2276 | .2444 | .680 | .806 | 883 |

- Main figure: Recall vs. Gini scatter, with the baseline marked and the Pareto frontier drawn.
- Second figure: the dial sweep. B's Gini goes up at β ≥ 0.5, from 0.830 to 0.854 to 0.907 on test. That reversal is a good visual.
- Reading the numbers:
  - Every configuration loses recall, so nothing dominates the baseline.
  - The exchange rate differs. B costs about 0.4 recall points per Gini point, and ABC about 0.9.
  - ABC has the lowest Gini and ARP, but it pays a 15% recall drop for a 5% Gini drop.
- Stage 2 is not finished (see "Status of the runs" above).

### 12. Discussion: did it work?
- Hypothesis: ABC gives a better trade-off. Verdict: partially. On split 0 it lowers bias the most but is not the most efficient, and it doesn't clearly beat B alone.
- What the data supports:
  - A did nothing on bias (Gini flat, Recall down 4–5%).
  - B helped only at β = 0.25.
  - C gave a small, cheap gain.
- Hypotheses to label as untested:
  - The bias may come mostly from the objective (sampling asymmetry) rather than the propagation. This would explain why A failed and B helped.
  - A's free ID embeddings and undiscounted layer 0 may let the model compensate.
  - B's reversal at high β may be the L2 confound. The normalized-B diagnostic was skipped.
- Caveats:
  - 200 epochs truncates the runs (7 of 8 were still improving).
  - There are only 2 splits.
  - The dials were tuned on split 0.
  - Metrics are at K = 20 only.

### 13. Future work
- Normalized-weight B, to test the L2 confound.
- Longer training.
- Head/mid/tail breakdown (the `recs/` files already support this with no extra GPU time).
- Popularity-aware negative sampling.
- Thresholded positives and low-rating negatives.
- True cold-start evaluation: only 22–31 items have zero train edges per split, too few to test C's claim.
- Larger γ and β ranges.

### 14. Questions

## Things to decide

1. Wait for the stage 2 split-1 runs? The combos on split 1 would make the results slide stronger.
2. Slide count and time limit? Once known, slides 7–9 can be merged into one or the discussion split.
3. Speaker notes written out per slide, in the Q&A style CLAUDE.md asks for?
