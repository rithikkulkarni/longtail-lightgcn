# 4-Triple Example: How Intervention B Re-weights BPR Training

The scores below are made up for illustration, and the numbers were chosen to make the effect easy to see. Do not present the percentages as experimental results.

## Background: what a "triple" is

BPR trains on triples `(u, i, j)`:
- `u` is a user.
- `i` is the **positive** item: one the user interacted with (a real training edge).
- `j` is the **negative** item: a random item the user has not interacted with in training. It is a loose label, since the user may simply not have seen it.

Each triple's loss is `-ln σ(pos − neg)`, which equals `ln(1 + e^-(pos − neg))`. It is near 0 when the positive scores well above the negative, `ln 2 ≈ 0.693` when they tie, and large when the negative wins.

An item is the positive once per training edge pointing at it, so a popular item is the positive in far more triples than a rare one.

## Intervention B

Each triple's loss is multiplied by `w(i) = pop(i)^-β`, where `i` is the **positive** item and `pop(i)` is its training degree (Eq. 2 of the proposal). The negative is not weighted. β = 0 gives weight 1 for every item, which is the baseline.

## Setup (β = 1)

| Item | Training degree | Weight `w = deg^-1` |
|---|---|---|
| A (popular) | 100 | 0.01 |
| B (medium) | 10 | 0.1 |
| C (rare) | 1 | 1.0 |

A is the positive in two of the four triples, because popular items have more edges and so appear as a positive more often. (In the real data this would be thousands of times more.)

## The four triples

"Gradient" is how hard the loss pushes the positive's score up: `w · σ(neg − pos) / 4`, where 4 is the batch size.

| # | Triple | pos score | neg score | Baseline loss | w | Weighted loss | Baseline gradient | Weighted gradient |
|---|---|---|---|---|---|---|---|---|
| 1 | (u1, **A**, X) | 2.0 | 0.0 | 0.1269 | 0.01 | 0.0013 | 0.0298 | 0.0003 |
| 2 | (u2, **A**, Y) | 1.0 | 0.5 | 0.4741 | 0.01 | 0.0047 | 0.0944 | 0.0009 |
| 3 | (u3, **B**, Z) | 0.0 | 0.0 | 0.6931 | 0.1 | 0.0693 | 0.1250 | 0.0125 |
| 4 | (u4, **C**, W) | −0.5 | 1.0 | 1.7014 | 1.0 | 1.7014 | 0.2044 | 0.2044 |
| | **Batch mean** | | | **0.7489** | | **0.4442** | | |

## Walking through it

- **Triple 1** is already right (A beats X by 2.0), so the baseline loss is small. B shrinks it by another 100x.
- **Triple 3** is a tie, so the loss is ln 2 ≈ 0.693. B gives it weight 0.1, so it contributes 0.069.
- **Triple 4** is wrong (the random negative W beats the rare positive C). It has the largest loss, and B leaves it at full strength.
- **The batch mean** drops from 0.749 to 0.444. The overall loss shrinks as β grows. This is a known side effect of un-normalized weights (see DESIGN.md D-27).

## Share of the total loss

| | Item A (2 triples) | Item B | Item C |
|---|---|---|---|
| Baseline | 20.1% | 23.1% | 56.8% |
| With B (β = 1) | 0.3% | 3.9% | 95.8% |

Under B the rare item's single triple dominates the training signal. This is the "equal gradient mass per item" idea. An item with degree d is the positive in about d triples per epoch, each weighted `d^-β`, so its total gradient mass is about `d^(1−β)`:
- At β = 0 it is proportional to popularity.
- At β = 1 it is about equal across items.

## A gentler setting: β = 0.5

The weights are A = 0.1, B = 0.316, C = 1.0. The weighted losses are 0.0127, 0.0474, 0.2192 and 1.7014, and the batch mean is 0.4952. The popular item is still discounted, but less aggressively.

## One sentence for a slide

*B scales each training triple's loss by `pop(i)^-β`, so a movie with 100 ratings pushes the model 100x less per triple than a movie with 1 rating at β = 1. That makes each movie's total influence roughly equal instead of proportional to popularity.*

## Checking the numbers

- `ln(1 + e^-2) = 0.1269`, `ln(1 + e^-0.5) = 0.4741`, `ln 2 = 0.6931`, `ln(1 + e^1.5) = 1.7014`.
- Baseline mean: (0.1269 + 0.4741 + 0.6931 + 1.7014) / 4 = 0.7489.
- Weighted mean (β = 1): (0.00127 + 0.00474 + 0.06931 + 1.70141) / 4 = 0.4442.
- Gradient on triple 1's positive: `0.01 · σ(−2) / 4 = 0.01 · 0.1192 / 4 ≈ 0.0003`.
