"""What a student misreads that its teacher reads correctly, so the next texts to record target real gaps."""

import unicodedata
from collections import Counter
from difflib import SequenceMatcher

import pandas as pd

from samleng.eval.cer import mixed_cer, normalize_khmer
from samleng.text import transliterate

COENG = "្"


def _spoken(text: str) -> str:
    return normalize_khmer(transliterate(text)).lower()


def clusters(text: str) -> list[str]:
    """Split into Khmer clusters: a consonant with its subscripts, vowel signs and marks."""
    out: list[str] = []
    for i, ch in enumerate(text):
        joined = i and (
            unicodedata.category(ch).startswith("M")
            or ch == COENG
            or text[i - 1] == COENG
        )
        if joined:
            out[-1] += ch
        else:
            out.append(ch)
    return out


def confusions(ref: str, hyp: str) -> list[tuple[str, str]]:
    """Each stretch of whole clusters the recognizer heard differently: (written, heard)."""
    a, b = clusters(_spoken(ref)), clusters(_spoken(hyp))
    ops = SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    return [
        ("".join(a[i1:i2]), "".join(b[j1:j2]))
        for op, i1, i2, j1, j2 in ops
        if op != "equal"
    ]


def worst_sentences(
    student: pd.DataFrame, teacher: pd.DataFrame, top: int = 25
) -> pd.DataFrame:
    """Sentences where the student's error is furthest above the teacher's."""
    df = student.merge(teacher[["id", "hyp"]], on="id", suffixes=("", "_teacher"))
    df["cer"] = [mixed_cer(r, h) for r, h in zip(df.ref, df.hyp, strict=True)]
    df["cer_teacher"] = [
        mixed_cer(r, h) for r, h in zip(df.ref, df.hyp_teacher, strict=True)
    ]
    df["gap"] = df.cer - df.cer_teacher
    cols = ["id", "category", "cer", "cer_teacher", "gap", "ref", "hyp"]
    return df.sort_values("gap", ascending=False).head(top)[cols]


def student_only_confusions(
    student: pd.DataFrame, teacher: pd.DataFrame, min_count: int = 2
) -> list[tuple[str, str, int, int]]:
    """(written, heard, sentences, teacher sentences) for misreadings the student makes more often than the teacher."""

    def count(df: pd.DataFrame) -> Counter:
        seen: Counter = Counter()
        for ref, hyp in zip(df.ref, df.hyp, strict=True):
            seen.update(set(confusions(ref, hyp)))
        return seen

    ours, theirs = count(student), count(teacher)
    rows = [
        (w, h, n, theirs[(w, h)])
        for (w, h), n in ours.items()
        if n >= min_count and n > theirs[(w, h)]
    ]
    return sorted(rows, key=lambda r: (r[3] - r[2], r[0]))


def hard_clusters(
    student: pd.DataFrame, teacher: pd.DataFrame, min_count: int = 3
) -> list[tuple[str, int, int]]:
    """(written cluster run, sentences misread by the student, by the teacher), whatever was heard instead."""

    def count(df: pd.DataFrame) -> Counter:
        seen: Counter = Counter()
        for ref, hyp in zip(df.ref, df.hyp, strict=True):
            seen.update({w for w, _ in confusions(ref, hyp) if w})
        return seen

    ours, theirs = count(student), count(teacher)
    rows = [
        (w, n, theirs[w]) for w, n in ours.items() if n >= min_count and n > theirs[w]
    ]
    return sorted(rows, key=lambda r: (r[2] - r[1], r[0]))
