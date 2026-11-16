from samleng.eval.gate import best_all

SUMMARY = """\
| system   | category |   n |   median_cer_base_pct |   median_cer_pct |   better |
|:---------|:---------|----:|----------------------:|-----------------:|---------:|
| step2000 | ALL      | 300 |                  10.5 |              9   |      147 |
| step6000 | ALL      | 300 |                  10.5 |              6.9 |      168 |
| step6000 | pure     |  80 |                   8.8 |              3.1 |       35 |
"""
PICK = """\
|          | ALL per seed    |   long |   pure |   ALL |
|:---------|:----------------|-------:|-------:|------:|
| step6000 | 9.4 / 8.6 / 9.5 |    4.2 |    7.5 |   8.8 |
| step8000 | 9.1 / 9.6 / 9.6 |    3.6 |    8.7 |   9.2 |

## Pairs (p = paired Wilcoxon)

| a        | b        | category |   n |   better |   worse |    p |
|:---------|:---------|:---------|----:|---------:|--------:|-----:|
| step6000 | step8000 | ALL      | 300 |      126 |     143 | 0.63 |
"""


def test_best_all_prefers_the_three_seed_pick(tmp_path):
    assert best_all(tmp_path) is None
    (tmp_path / "summary.md").write_text(SUMMARY)
    assert best_all(tmp_path) == (6.9, "single-seed summary")
    (tmp_path / "pick.md").write_text(PICK)
    assert best_all(tmp_path) == (8.8, "3-seed pick")
