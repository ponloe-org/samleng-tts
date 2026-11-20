import pandas as pd

from samleng.eval.ladder import result_files, verdict


def frame(cers):
    return pd.DataFrame(
        {"id": [f"s{i}" for i in range(len(cers))], "category": "pure", "cer": cers}
    )


def test_challenger_must_be_clearly_and_significantly_better():
    champ = frame([0.10] * 40)
    assert verdict(champ, frame([0.05] * 40))["win"]
    # Better, but by less than half a point.
    assert not verdict(champ, frame([0.097] * 40))["win"]
    # Lower median and more sentences better, but much worse on the rest.
    assert not verdict(champ, frame([0.09] * 21 + [0.30] * 19))["win"]
    # Lower median from a handful of sentences: not significant.
    assert not verdict(champ, frame([0.10] * 17 + [0.09] * 3 + [0.0] + [0.10] * 19))[
        "win"
    ]
    assert not verdict(champ, frame([0.12] * 40))["win"]


def test_result_files_follow_the_picker_layout(tmp_path):
    files = result_files(tmp_path, "run", [6000, 8000], translit=True)
    assert files[0] == tmp_path / "run_translit" / "6000" / "results.csv"
    assert files[2] == tmp_path / "run_translit_seed2" / "6000" / "results.csv"
    assert len(files) == 6
