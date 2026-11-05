import json
import math

import numpy as np
import pytest
import torch

from longtail_lightgcn.data.preprocess import make_split, save_split
from longtail_lightgcn.metrics import (
    all_metrics, average_recommendation_popularity, coverage, gini, item_exposure, ndcg_at_k, recall_at_k,
    target_mask, target_popularity,
)
from longtail_lightgcn.run_grid import (
    DIALS, RunSpec, done_keys, main, read_rows, stage1_specs, stage2_specs,
)
from longtail_lightgcn.select_dials import select_dials
from longtail_lightgcn.select_dials import main as select_main


# ---- the grid itself (D-29) ----

def test_stage1_is_eight_runs_on_split_zero():
    specs = stage1_specs()
    assert len(specs) == 8 and {s.split for s in specs} == {0}
    assert sorted(s.label for s in specs) == ["A", "A", "A", "B", "B", "B", "C", "baseline"]
    assert len({s.key for s in specs}) == 8


def test_stage2_is_the_eight_combinations_on_two_splits():
    specs = stage2_specs(0.5, 0.25)
    assert len(specs) == 16 and len({s.key for s in specs}) == 16
    assert sorted({s.label for s in specs}) == sorted(["baseline", "A", "B", "C", "AB", "AC", "BC", "ABC"])
    abc = [s for s in specs if s.label == "ABC"][0]
    assert (abc.gamma, abc.beta, abc.use_content) == (0.5, 0.25, True)


def test_stage_overlap_gives_twenty_runs_in_total():
    # baseline, A, B, C on split 0 are shared, so the whole grid is 8 + 16 - 4 = 20 distinct runs
    keys = {s.key for s in stage1_specs()} | {s.key for s in stage2_specs(0.5, 0.25)}
    assert len(keys) == 20


def test_shared_runs_are_skipped_once_stage1_is_done():
    rows = [{"key": s.key, "config": {"max_epochs": 200}} for s in stage1_specs()]
    done = done_keys(rows)
    assert len([s for s in stage2_specs(0.5, 0.25) if s.key not in done]) == 12
    assert len([s for s in stage2_specs(0.25, 0.75) if s.key not in done]) == 12  # any dial in DIALS is shared


def test_smoke_budget_rows_do_not_count_as_done():
    rows = [{"key": "k", "config": {"max_epochs": 2}}]
    assert done_keys(rows, budget=200) == set() and done_keys(rows, budget=2) == {"k"}


# ---- end to end on a tiny dataset ----

@pytest.fixture
def tiny(tmp_path):
    rng = np.random.default_rng(0)
    n_users, n_items = 40, 60
    users, items = [], []
    for u in range(n_users):
        picked = rng.choice(n_items, size=25, replace=False)
        users += [u] * 25
        items += list(picked)
    for s in (0, 1):
        save_split(tmp_path / f"split_{s}.npz", make_split(np.array(users), np.array(items), seed=s), n_users, n_items)
    rows = ["movie_id,item_idx,title,genres,in_catalog"]
    rows += [f"{i + 1},{i},Movie {i} ({1950 + i}),{'Comedy' if i % 2 else 'Drama|War'},True" for i in range(n_items)]
    (tmp_path / "items.csv").write_text("\n".join(rows) + "\n")
    return tmp_path, n_users, n_items


def grid(tiny, *extra):
    d = tiny[0]
    return main(["--results_dir", str(d / "res"), "--data_dir", str(d), "--device", "cpu",
                 "--max_epochs", "2", *extra])


def test_one_run_writes_a_complete_row_and_recs(tiny):
    d, n_users, n_items = tiny
    written = grid(tiny, "--stage", "1")
    assert len(written) == 1  # the default is one run per session
    row = read_rows(d / "res" / "results.jsonl")[0]
    assert row["key"] == "g0_b0_c0_s0" and row["label"] == "baseline"
    assert row["config"]["max_epochs"] == 2 and row["config"]["split_seed"] == row["config"]["seed"] == 0
    for part in ("val", "test"):
        assert set(row[part]) == {"recall@20", "ndcg@20", "coverage", "gini", "arp", "target_pop"}
        assert all(math.isfinite(v) for v in row[part].values())
        assert 0 <= row[part]["recall@20"] <= 1 and 0 <= row[part]["gini"] < 1
    assert row["history"][-1]["epoch"] == 2 and row["git_commit"] is not None
    recs = np.load(d / "res" / "recs" / "g0_b0_c0_s0.npz")["recs"]
    assert recs.shape == (n_users, 20) and recs.dtype == np.int16 and recs.max() < n_items


def test_repeated_sessions_take_the_next_run_and_never_repeat(tiny):
    d = tiny[0]
    first = grid(tiny, "--stage", "1")[0]["key"]
    second = grid(tiny, "--stage", "1")[0]["key"]
    assert first != second
    keys = [r["key"] for r in read_rows(d / "res" / "results.jsonl")]
    assert keys == [first, second]
    assert grid(tiny, "--stage", "1", "--run_id", first) == []  # already done: nothing is rerun


def test_run_id_runs_that_specific_run_including_content(tiny):
    written = grid(tiny, "--stage", "1", "--run_id", "g0_b0_c1_s0")
    assert [r["label"] for r in written] == ["C"]
    assert "content_mlp" not in json.dumps(written[0]["config"])  # config holds flags, not weights
    assert written[0]["config"]["use_content"] is True


def test_max_runs_and_max_hours(tiny):
    assert len(grid(tiny, "--stage", "1", "--max_runs", "2")) == 2
    # a zero-hour limit lets the first run start (nothing to judge by yet) and then stops
    assert len(grid(tiny, "--stage", "1", "--max_runs", "3", "--max_hours", "0")) == 1


def test_list_does_not_run_anything(tiny, capsys):
    assert grid(tiny, "--stage", "1", "--list") == []
    out = capsys.readouterr().out
    assert out.count("pending") == 8 and "0 of 8 runs done" in out
    assert not (tiny[0] / "res" / "results.jsonl").exists()


def test_stage2_needs_dials_json_and_uses_it(tiny, capsys):
    with pytest.raises(SystemExit):
        grid(tiny, "--stage", "2")
    res = tiny[0] / "res"
    res.mkdir()
    (res / "dials.json").write_text(json.dumps({"gamma": 0.5, "beta": 0.25}))
    grid(tiny, "--stage", "2", "--list")
    out = capsys.readouterr().out
    assert out.count("pending") == 16 and "g0.5_b0.25_c1_s1" in out  # ABC on split 1


def test_stage2_reuses_finished_stage1_runs(tiny, capsys):
    res = tiny[0] / "res"
    res.mkdir()
    (res / "dials.json").write_text(json.dumps({"gamma": 0.5, "beta": 0.25}))
    fake = [{"key": s.key, "config": {"max_epochs": 2}} for s in stage1_specs() if s.gamma in (0, 0.5) and s.beta in (0, 0.25)]
    (res / "results.jsonl").write_text("\n".join(json.dumps(r) for r in fake) + "\n")
    grid(tiny, "--stage", "2", "--list")
    assert "4 of 16 runs done" in capsys.readouterr().out  # baseline, A, B, C on split 0


def test_unreadable_line_is_skipped_not_fatal(tiny):
    res = tiny[0] / "res"
    res.mkdir()
    (res / "results.jsonl").write_text('{"key": "x", "config": {"max_epochs": 2}}\n{"key": "tor')
    assert [r["key"] for r in read_rows(res / "results.jsonl")] == ["x"]


# ---- all_metrics ----

def test_all_metrics_matches_the_individual_functions():
    recs = torch.tensor([[0, 1, 2], [1, 0, 4], [2, 3, 4]])
    target = target_mask([(np.array([0, 0, 1]), np.array([1, 4, 0]))], 3, 6)
    degree = torch.tensor([10, 0, 5, 2, 1, 7])
    m = all_metrics(recs, target, degree, 6)
    assert m["recall@3"] == recall_at_k(recs, target) and m["ndcg@3"] == ndcg_at_k(recs, target)
    assert m["coverage"] == coverage(recs, 6) and m["gini"] == gini(item_exposure(recs, 6))
    assert m["arp"] == average_recommendation_popularity(recs, degree)
    assert m["target_pop"] == target_popularity(target, degree)


# ---- dial selection (D-29) ----

def fake_rows(a, b, base=(0.25, 0.9), budget=200):
    """a, b: {dial: (val_recall, val_gini)}; base: baseline (val_recall, val_gini)."""
    def row(spec, vals):
        return {"key": spec.key, "config": {"max_epochs": budget}, "val": {"recall@20": vals[0], "gini": vals[1]}}
    rows = [row(RunSpec(), base), row(RunSpec(use_content=True), base)]
    rows += [row(RunSpec(gamma=d), a[d]) for d in DIALS] + [row(RunSpec(beta=d), b[d]) for d in DIALS]
    return rows


def test_rule_picks_lowest_gini_within_tolerance():
    a = {0.25: (0.245, 0.88), 0.5: (0.240, 0.80), 0.75: (0.200, 0.60)}  # 0.75 is lowest Gini but loses 20% Recall
    b = {0.25: (0.249, 0.89), 0.5: (0.248, 0.85), 0.75: (0.2375, 0.82)}  # 0.2375 is exactly 95% of 0.25: allowed
    out = select_dials(fake_rows(a, b))
    assert out["gamma"] == 0.5 and out["beta"] == 0.75
    assert out["fallback"] == {"gamma": False, "beta": False}
    assert [t["within_tolerance"] for t in out["table"]["gamma"]] == [True, True, False]


def test_fallback_to_the_weakest_dial_when_nothing_qualifies():
    bad = {d: (0.10, 0.5) for d in DIALS}
    ok = {d: (0.25, 0.9 - d / 10) for d in DIALS}
    out = select_dials(fake_rows(bad, ok))
    assert out["gamma"] == 0.25 and out["fallback"]["gamma"] is True
    assert out["beta"] == 0.75 and out["fallback"]["beta"] is False


def test_tolerance_changes_the_choice():
    a = {0.25: (0.245, 0.8), 0.5: (0.235, 0.7), 0.75: (0.200, 0.5)}
    same = {d: (0.25, 0.8) for d in DIALS}
    assert select_dials(fake_rows(a, same), tolerance=0.01)["gamma"] == 0.25  # floor 0.2475: nothing qualifies -> fallback
    assert select_dials(fake_rows(a, same), tolerance=0.03)["gamma"] == 0.25  # floor 0.2425: only 0.25
    assert select_dials(fake_rows(a, same), tolerance=0.07)["gamma"] == 0.5  # floor 0.2325: 0.25 and 0.5, lower Gini wins
    assert select_dials(fake_rows(a, same), tolerance=0.25)["gamma"] == 0.75  # floor 0.1875: all qualify


def test_ties_on_gini_go_to_the_smaller_dial():
    tie = {0.25: (0.25, 0.8), 0.5: (0.24, 0.7), 0.75: (0.24, 0.7)}  # 0.5 and 0.75 tie on Gini
    flat = {d: (0.25, 0.8) for d in DIALS}  # everything ties
    out = select_dials(fake_rows(tie, flat))
    assert out["gamma"] == 0.5 and out["beta"] == 0.25


def test_incomplete_stage1_or_smoke_rows_raise():
    a = b = {d: (0.25, 0.8) for d in DIALS}
    with pytest.raises(ValueError):
        select_dials(fake_rows(a, b)[:-1])
    with pytest.raises(ValueError):
        select_dials(fake_rows(a, b, budget=2))  # rows from a smoke budget are ignored, so nothing is found


def test_select_dials_only_reads_validation_numbers():
    a = b = {d: (0.25, 0.8) for d in DIALS}
    rows = fake_rows(a, b)
    for r in rows:
        r["test"] = {"recall@20": float("nan"), "gini": float("nan")}  # would poison any use of test
    assert select_dials(rows)["gamma"] == 0.25


def test_select_dials_writes_dials_json(tmp_path):
    a = {0.25: (0.245, 0.88), 0.5: (0.240, 0.80), 0.75: (0.200, 0.60)}
    b = {d: (0.25, 0.9 - d / 10) for d in DIALS}
    (tmp_path / "results.jsonl").write_text("\n".join(json.dumps(r) for r in fake_rows(a, b)) + "\n")
    select_main(["--results_dir", str(tmp_path)])
    saved = json.loads((tmp_path / "dials.json").read_text())
    assert saved["gamma"] == 0.5 and saved["beta"] == 0.75
    select_main(["--results_dir", str(tmp_path), "--no_write", "--tolerance", "0.5"])  # smoke: does not overwrite
    assert json.loads((tmp_path / "dials.json").read_text())["gamma"] == 0.5
