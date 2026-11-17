import pandas as pd

from samleng.eval.errors import (
    confusions,
    hard_clusters,
    student_only_confusions,
    worst_sentences,
)


def frame(rows):
    return pd.DataFrame(
        [{"id": i, "category": "pure", "ref": r, "hyp": h} for i, r, h in rows]
    )


def test_confusions_cover_whole_clusters():
    assert confusions("ខ្យល់បក់ ។", "ខ្យល់បក់") == []
    # Only the base consonant differs, but the subscript and vowel belong to it.
    assert confusions("ខ្យល់បក់", "ខ្ជិលបក់") == [("ខ្យល់", "ខ្ជិល")]
    assert confusions("សែសិប", "ខ្សែសិប") == [("សែ", "ខ្សែ")]


def test_reports_only_what_the_teacher_gets_right():
    teacher = frame(
        [
            ("a", "មេឃស្រឡះ", "មេឃស្រឡះ"),
            ("b", "កូនៗរៀន", "កូនកោររៀន"),
            ("c", "មេឃល្អ", "មេឃល្អ"),
        ]
    )
    student = frame(
        [
            ("a", "មេឃស្រឡះ", "នេះស្រឡះ"),
            ("b", "កូនៗរៀន", "កូនកោររៀន"),
            ("c", "មេឃល្អ", "នេះល្អ"),
        ]
    )
    assert student_only_confusions(student, teacher) == [("មេឃ", "នេះ", 2, 0)]
    assert hard_clusters(student, teacher, min_count=2) == [("មេឃ", 2, 0)]
    worst = worst_sentences(student, teacher, top=2)
    assert set(worst.id) == {"a", "c"}
    assert (worst.cer_teacher == 0).all()
