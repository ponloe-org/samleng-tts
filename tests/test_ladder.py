import pandas as pd

from samleng.eval.ladder import pooled, result_files, verdict


def frame(cers, category="pure"):
    return pd.DataFrame(
        {"id": [f"s{i}" for i in range(len(cers))], "category": category, "cer": cers}
    )


def model(cers, seconds=100.0, latin=None):
    f = frame(cers)
    return {"pooled": f, "latin": latin if latin is not None else f, "seconds": seconds}


def test_challenger_must_be_clearly_and_significantly_better():
    champ = model([0.10] * 40)
    assert verdict(champ, model([0.05] * 40))["win"]
    # Better, but by less than half a point.
    assert not verdict(champ, model([0.097] * 40))["win"]
    # Lower on most sentences, but much worse on the rest: the mean says no.
    assert not verdict(champ, model([0.09] * 21 + [0.30] * 19))["win"]
    assert not verdict(champ, model([0.12] * 40))["win"]


def test_a_slower_or_category_breaking_challenger_is_blocked():
    champ = model([0.10] * 40)
    assert verdict(champ, model([0.05] * 40, seconds=90.0))["win"]
    slow = verdict(champ, model([0.05] * 40, seconds=110.0))
    assert not slow["win"] and "slower" in slow["blocks"][0]
    mixed = frame([0.10] * 40, "mixed_unseen")
    worse_english = verdict(
        model([0.10] * 40, latin=mixed), model([0.05] * 40, latin=mixed.assign(cer=0.2))
    )
    assert not worse_english["win"] and "mixed_unseen" in worse_english["blocks"][0]


def test_pooled_averages_inputs_only_for_mixed_sentences():
    latin = pd.DataFrame(
        {"id": ["a", "b"], "category": ["pure", "mixed_seen"], "cer": [0.2, 0.4]}
    )
    khmer = latin.assign(cer=[0.9, 0.0])
    assert pooled(latin, khmer)["cer"].tolist() == [0.2, 0.2]


def test_result_files_follow_the_picker_layout(tmp_path):
    files = result_files(tmp_path, "run", [6000, 8000], translit=True)
    assert files[0] == tmp_path / "run_translit" / "6000" / "results.csv"
    assert files[2] == tmp_path / "run_translit_seed2" / "6000" / "results.csv"
    assert len(files) == 6
    fresh = result_files(tmp_path, "run", [6000], translit=False, seeds=(3, 4, 5))
    assert fresh[0] == tmp_path / "run_seed3" / "6000" / "results.csv"
